#!/usr/bin/env python
#
# clcache.py - a compiler cache for Microsoft Visual Studio
#
# Copyright (c) 2010, 2011, 2012, 2013 froglogic GmbH <raabe@froglogic.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
from ctypes import windll, wintypes
import codecs
from collections import defaultdict, namedtuple
import cPickle as pickle
import hashlib
import json
import os
from shutil import copyfile, rmtree
import subprocess
from subprocess import Popen, PIPE, STDOUT
import sys
import struct
import tempfile
import multiprocessing
import re

HASH_ALGORITHM = hashlib.sha1

# Manifest file will have at most this number of hash lists in it. Need to avoi
# manifests grow too large.
MAX_MANIFEST_HASHES = 100

# String, by which BASE_DIR will be replaced in paths, stored in manifests.
# ? is invalid character for file name, so it seems ok
# to use it as mark for relative path.
BASEDIR_REPLACEMENT = '?'

# Size of buffers or pipes, used by daemons.
PIPE_BUFFER_SIZE = 1024

# includeFiles - list of paths toi include files, which this source file use.
# hashes - dictionary.
# Key - cumulative hash of all include files in includeFiles;
# Value - key in the cache, under which output file is stored.
Manifest = namedtuple('Manifest', ['includeFiles', 'hashes'])

# It is expected that during building we have only few possible
# PATH variants, so caching should work great here in dameon mode.
COMPILER_PATH_CACHE = {}
VERIFIED_COMPILER_HINTS = set()

# Many source files will include the same includes (e.g system includes).
# We will cache their hashes based on path and file modification time, to
# speed up build process.
HEADER_HASH_CACHE = {}

# When clearing objects from cache we need to remove also orphaned manifests.
# Without this manifest count will grow infinitely consuming disk space
# and slowing further cache clearing.
# So, we add empty "mark" files to the directories with cache entries.
# By names of these "mark" files we can determine cache key of corresponding
# manifest and update/remove it during cache clearing.
MANIFEST_MARK_EXTENSION = '.mark'

# Path - either absolute or relative to BASE_DIR (if possible) path,
# mtime - modification time of file. We need use it since some headers may be
# re-generated during build process.
def get_header_key(path, mtime):
    return '{mtime}:{path}'.format(mtime=mtime, path=path)

class ObjectCacheLockException(Exception):
    pass

class LogicException(Exception):
    def __init__(self, message):
        self.value = message

    def __str__(self):
        return repr(self.message)

class ObjectCacheLock:
    """ Implements a lock for the object cache which
    can be used in 'with' statements. """
    INFINITE = 0xFFFFFFFF

    def __init__(self, mutexName, timeoutMs):
        mutexName = 'Local\\' + mutexName
        self._mutex = windll.kernel32.CreateMutexW(
            wintypes.c_int(0),
            wintypes.c_int(0),
            unicode(mutexName))
        self._timeoutMs = timeoutMs
        # We use this class only for inter-process synchronization, so
        # we can use local variable here to avoid too many calls of system APIs
        self._acquire_count = 0
        assert self._mutex

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()

    def __del__(self):
        windll.kernel32.CloseHandle(self._mutex)

    def acquire(self):
        self._acquire_count += 1
        if self._acquire_count > 1:
            return
        WAIT_ABANDONED = 0x00000080
        result = windll.kernel32.WaitForSingleObject(
            self._mutex, wintypes.c_int(self._timeoutMs))
        if result != 0 and result != WAIT_ABANDONED:
            errorString ='Error! WaitForSingleObject returns {result}, last error {error}'.format(
                result=result,
                error=windll.kernel32.GetLastError())
            raise ObjectCacheLockException(errorString)

    def release(self):
        self._acquire_count -= 1
        if self._acquire_count > 0:
            return
        windll.kernel32.ReleaseMutex(self._mutex)

class ObjectCache:
    def __init__(self):
        try:
            self.dir = os.environ["CLCACHE_DIR"]
        except KeyError:
            self.dir = os.path.join(os.path.expanduser("~"), "clcache")
        lockName = self.cacheDirectory().replace(':', '-').replace('\\', '-')
        self.lock = ObjectCacheLock(lockName, ObjectCacheLock.INFINITE)

        self.tempDir = os.path.join(self.dir, '.temp')
        self.daemonsDir = os.path.join(self.dir, '.daemons')
        self.manifestsDir = os.path.join(self.dir, "manifests")
        self.objectsDir = os.path.join(self.dir, "objects")

        # Creates both self.dir and self.tempDir if neccessary
        if (not (os.path.exists(self.tempDir)) or
            not (os.path.exists(self.daemonsDir)) or
            not (os.path.exists(self.manifestsDir)) or
            not (os.path.exists(self.objectsDir))):
        # Guarded by lock to avoid exceptions when multiple processes started
        # and try to create the same dir.
            with self.lock:
                if not os.path.exists(self.tempDir):
                    os.makedirs(self.tempDir)
                if not os.path.exists(self.daemonsDir):
                    os.makedirs(self.daemonsDir)
                if not os.path.exists(self.manifestsDir):
                    os.makedirs(self.manifestsDir)
                if not os.path.exists(self.objectsDir):
                    os.makedirs(self.objectsDir)

    def cacheDirectory(self):
        return self.dir

    def clean(self, stats, maximumSize):
        with self.lock:
            currentSize = stats.currentCacheSize()
            if currentSize < maximumSize:
                return
            currentEntriesCount = stats.numCacheEntries()

            # Free at least 10% to avoid cleaning up too often which
            # is a big performance hit with large caches.
            effectiveMaximumSize = maximumSize * 0.9

            objects = [os.path.join(root, "object")
                       for root, folder, files in os.walk(self.objectsDir)
                       if "object" in files]

            objectInfos = [(os.stat(fn), fn) for fn in objects]
            objectInfos.sort(key=lambda t: t[0].st_atime)

            for stat, fn in objectInfos:
                entryDir = os.path.split(fn)[0]
                entryHash = os.path.basename(entryDir)
                # Find all manifests and remove link from them
                for file in os.listdir(entryDir):
                    nameBase, ext = os.path.splitext(file)
                    if ext != MANIFEST_MARK_EXTENSION:
                        continue
                    self._removeEntryFromManifest(nameBase, entryHash, stats)
                rmtree(entryDir)
                currentEntriesCount -= 1
                currentSize -= stat.st_size
                if currentSize < effectiveMaximumSize:
                    break
            stats.setCacheSize(currentSize, currentEntriesCount)

    def _removeEntryFromManifest(self, manifestHash, entryHash, stats):
        manifest = self.getManifest(manifestHash)
        if not manifest:
            return
        for keyInManifest, cacheKey in manifest.hashes.items():
            if cacheKey == entryHash:
                del manifest.hashes[keyInManifest]
        if len(manifest.hashes) == 0:
            self.removeManifest(manifestHash)
            stats.removeManifest()
        else:
            self.setManifest(manifestHash, manifest)

    def removeObjects(self, stats, removedObjects):
        if len(removedObjects) == 0:
            return
        with self.lock:
            currentSize = stats.currentCacheSize()
            currentEntriesCount = stats.numCacheEntries()
            for hash in removedObjects:
                dirPath = self._cacheEntryDir(hash)
                if not os.path.exists(dirPath):
                    continue  # May be if object already evicted.
                objectPath = os.path.join(dirPath, "object")
                if os.path.exists(objectPath):
                    # May be absent if this if cached compiler
                    # output (for preprocess-only).
                    fileStat = os.stat(objectPath)
                    currentSize -= fileStat.st_size
                rmtree(dirPath)
                currentEntriesCount -= 1
            stats.setCacheSize(currentSize, currentEntriesCount)

    def getManifestHash(self, compilerBinary, commandLine, sourceFile):
        stat = os.stat(compilerBinary)
        # NOTE: We intentionally do not normalize command line to include
        # preprocessor options. In direct mode we do not perform
        # preprocessing before cache lookup, so all parameters are important
        additionalData = '{mtime}{size}{cmdLine}'.format(
            mtime=stat.st_mtime,
            size=stat.st_size,
            cmdLine=' '.join(commandLine));
        return getFileHash(sourceFile, additionalData)

    def computeKey(self, compilerBinary, commandLine):
        ppcmd = [compilerBinary, "/EP"]
        ppcmd += [arg for arg in commandLine if not arg in ("-c", "/c")]
        preprocessor = Popen(ppcmd, stdout=PIPE, stderr=PIPE)
        (preprocessedSourceCode, pperr) = preprocessor.communicate()

        if preprocessor.returncode != 0:
            sys.stderr.write(pperr)
            sys.stderr.write("clcache: preprocessor failed\n")
            sys.exit(preprocessor.returncode)

        normalizedCmdLine = self._normalizedCommandLine(commandLine)

        stat = os.stat(compilerBinary)
        h = HASH_ALGORITHM()
        h.update(str(stat.st_mtime))
        h.update(str(stat.st_size))
        h.update(' '.join(normalizedCmdLine))
        h.update(preprocessedSourceCode)
        return h.hexdigest()

    def getKeyInManifest(self, listOfHeaderHashes):
        return getHash(','.join(listOfHeaderHashes))

    def getDirectCacheKey(self, manifestHash, keyInManifest):
        # We must take into account manifestHash to avoid
        # collisions when different source files use the same
        # set of includes.
        return getHash(manifestHash + keyInManifest)

    def hasEntry(self, key):
        with self.lock:
            objectFileName = self.cachedObjectName(key)
            if os.path.exists(objectFileName):
                # Sometimes empty .obj files appears in cache (e.g. if computer is incorrectly
                # turned off during build process). Do not use these files, since they will fail
                # the build. They will be evicted normally during some cache clean.
                return os.path.getsize(objectFileName) > 0
            # If there are no .obj file, it may appear that we just cached compiler output
            # (e.g. if this is cached preprocessor invocation).
            return os.path.exists(self._cachedCompilerOutputName(key))

    def setEntry(self, key, objectFileName, compilerOutput, compilerStderr, manifestHash):
        with self.lock:
            if not os.path.exists(self._cacheEntryDir(key)):
                os.makedirs(self._cacheEntryDir(key))
            if objectFileName != '':
                copyOrLink(objectFileName, self.cachedObjectName(key))
            open(self._cachedCompilerOutputName(key), 'w').write(compilerOutput)
            if compilerStderr != '':
                open(self._cachedCompilerStderrName(key), 'w').write(compilerStderr)
            if manifestHash:
                # Save hash of the parent manifest to ensure reference will
                # be removed from it during cache cleaning.
                manifestMarkFileName = os.path.join(
                        self._cacheEntryDir(key),
                        manifestHash + MANIFEST_MARK_EXTENSION)
                open(manifestMarkFileName, 'w').close()

    # Returns true if this is new manifest
    def setManifest(self, manifestHash, manifest):
        with self.lock:
            if not os.path.exists(self._manifestDir(manifestHash)):
                os.makedirs(self._manifestDir(manifestHash))
            fileName = self._manifestName(manifestHash)
            result = not os.path.exists(fileName)
            with open(fileName, 'wb') as outFile:
                pickle.dump(manifest, outFile)
            return result

    def removeManifest(self, manifestHash):
        with self.lock:
            fileName = self._manifestName(manifestHash)
            if os.path.exists(fileName):
                os.remove(fileName)

    def getManifest(self, manifestHash):
        with self.lock:
            fileName = self._manifestName(manifestHash)
            if not os.path.exists(fileName):
                return None
            with open(fileName, 'rb') as inFile:
                try:
                    return pickle.load(inFile)
                except:
                    # Seems, file is corrupted
                    return None

    def cachedObjectName(self, key):
        return os.path.join(self._cacheEntryDir(key), "object")

    def cachedCompilerOutput(self, key):
        return open(self._cachedCompilerOutputName(key), 'r').read()

    def cachedCompilerStderr(self, key):
        fileName = self._cachedCompilerStderrName(key)
        if os.path.exists(fileName):
            return open(fileName, 'r').read()
        return ''

    def getTempFilePath(self, sourceFile):
        ext =  os.path.splitext(sourceFile)[1]
        handle, path = tempfile.mkstemp(suffix=ext, dir=self.tempDir)
        os.close(handle)
        return path

    def getDaemonDir(self, daemonPid):
        return os.path.join(self.daemonsDir, str(daemonPid))

    def regiterDaemon(self, daemonPid):
        with self.lock:
            os.makedirs(self.getDaemonDir(daemonPid))

    def unregiterDaemon(self, daemonPid):
        with self.lock:
            daemonDir = self.getDaemonDir(daemonPid)
            print 'DAEMON STDOUT:'
            with open(os.path.join(daemonDir, 'stdout.txt'), 'r') as f:
                sys.stdout.write(f.read())
            print 'DAEMON STDERR:'
            with open(os.path.join(daemonDir, 'stderr.txt'), 'r') as f:
                sys.stdout.write(f.read())
            rmtree(daemonDir)

    def getAllDaemonPids(self):
        with self.lock:
            dirs = os.listdir(self.daemonsDir)
        return [int(d) for d in dirs]

    def _cacheEntryDir(self, key):
        return os.path.join(self.objectsDir, key[:2], key)

    def _manifestDir(self, manifestHash):
        return os.path.join(self.manifestsDir, manifestHash[:2])

    def _manifestName(self, manifestHash):
        return os.path.join(self._manifestDir(manifestHash), manifestHash + ".dat")

    def _cachedCompilerOutputName(self, key):
        return os.path.join(self._cacheEntryDir(key), "output.txt")

    def _cachedCompilerStderrName(self, key):
        return os.path.join(self._cacheEntryDir(key), "stderr.txt")

    def _normalizedCommandLine(self, cmdline):
        # Remove all arguments from the command line which only influence the
        # preprocessor; the preprocessor's output is already included into the
        # hash sum so we don't have to care about these switches in the
        # command line as well.
        _argsToStrip = ("AI", "C", "E", "P", "FI", "u", "X",
                        "FU", "D", "EP", "Fx", "U", "I")

        # Also remove the switch for specifying the output file name; we don't
        # want two invocations which are identical except for the output file
        # name to be treated differently.
        _argsToStrip += ("Fo",)

        return [arg for arg in cmdline
                if not (arg[0] in "/-" and arg[1:].startswith(_argsToStrip))]

class PersistentJSONDict:
    def __init__(self, fileName):
        self._dirty = False
        self._dict = {}
        self._fileName = fileName
        try:
            self._dict = json.load(open(self._fileName, 'r'))
        except:
            pass

    def save(self):
        if self._dirty:
            json.dump(self._dict, open(self._fileName, 'w'))

    def __setitem__(self, key, value):
        self._dict[key] = value
        self._dirty = True

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict


class Configuration:
    _defaultValues = { "MaximumCacheSize": 1024 * 1024 * 1000 }

    def __init__(self, objectCache):
        self._objectCache = objectCache
        with objectCache.lock:
            self._cfg = PersistentJSONDict(os.path.join(objectCache.cacheDirectory(),
                                                        "config.txt"))
        for setting, defaultValue in self._defaultValues.iteritems():
            if not setting in self._cfg:
                self._cfg[setting] = defaultValue

    def maximumCacheSize(self):
        return self._cfg["MaximumCacheSize"]

    def setMaximumCacheSize(self, size):
        self._cfg["MaximumCacheSize"] = size

    def save(self):
        with self._objectCache.lock:
            self._cfg.save()


class CacheStatistics:
    def __init__(self, objectCache):
        # Use two dictionaries to ensure we'll grab cache lock on the smallest
        # possible time. We collect increment _incremental_stats while possible
        # and then merge it with stats on disk.
        self._incremental_stats = defaultdict(int)
        self._stats = None
        self._objectCache = objectCache

    def numCallsWithoutSourceFile(self):
        self.ensureLoadedAndLocked()
        return self._stats["CallsWithoutSourceFile"]

    def registerCallWithoutSourceFile(self):
        stats = self._stats if self._stats else self._incremental_stats
        stats["CallsWithoutSourceFile"] += 1

    def numCallsWithMultipleSourceFiles(self):
        self.ensureLoadedAndLocked()
        return self._stats["CallsWithMultipleSourceFiles"]

    def registerCallWithMultipleSourceFiles(self):
        stats = self._stats if self._stats else self._incremental_stats
        stats["CallsWithMultipleSourceFiles"] += 1

    def numCallsWithPch(self):
        self.ensureLoadedAndLocked()
        return self._stats["CallsWithPch"]

    def registerCallWithPch(self):
        stats = self._stats if self._stats else self._incremental_stats
        stats["CallsWithPch"] += 1

    def numCallsForLinking(self):
        self.ensureLoadedAndLocked()
        return self._stats["CallsForLinking"]

    def registerCallForLinking(self):
        stats = self._stats if self._stats else self._incremental_stats
        stats["CallsForLinking"] += 1

    def numEvictedMisses(self):
        self.ensureLoadedAndLocked()
        return self._stats["EvictedMisses"]

    def registerEvictedMiss(self):
        self.registerCacheMiss()
        stats = self._stats if self._stats else self._incremental_stats
        stats["EvictedMisses"] += 1

    def numHeaderChangedMisses(self):
        self.ensureLoadedAndLocked()
        return self._stats["HeaderChangedMisses"]

    def registerHeaderChangedMiss(self):
        self.registerCacheMiss()
        stats = self._stats if self._stats else self._incremental_stats
        stats["HeaderChangedMisses"] += 1

    def numSourceChangedMisses(self):
        return self._stats["SourceChangedMisses"]

    def registerSourceChangedMiss(self):
        self.registerCacheMiss()
        stats = self._stats if self._stats else self._incremental_stats
        stats["SourceChangedMisses"] += 1

    def numCacheEntries(self):
        self.ensureLoadedAndLocked()
        return self._stats["CacheEntries"]

    def registerCacheEntry(self, size):
        stats = self._stats if self._stats else self._incremental_stats
        stats["CacheEntries"] += 1
        stats["CacheSize"] += size

    def numManifests(self):
        self.ensureLoadedAndLocked()
        return self._stats["ManifestsCount"]

    def registerManifest(self):
        stats = self._stats if self._stats else self._incremental_stats
        stats["ManifestsCount"] += 1

    def removeManifest(self):
        stats = self._stats if self._stats else self._incremental_stats
        stats["ManifestsCount"] -= 1

    def currentCacheSize(self):
        self.ensureLoadedAndLocked()
        return self._stats["CacheSize"]

    def setCacheSize(self, size, entriesCount):
        self.ensureLoadedAndLocked()
        self._stats["CacheSize"] = size
        self._stats["CacheEntries"] = entriesCount

    def numCacheHits(self):
        self.ensureLoadedAndLocked()
        return self._stats["CacheHits"]

    def registerCacheHit(self):
        stats = self._stats if self._stats else self._incremental_stats
        stats["CacheHits"] += 1

    def numCacheMisses(self):
        self.ensureLoadedAndLocked()
        return self._stats["CacheMisses"]

    def registerCacheMiss(self):
        stats = self._stats if self._stats else self._incremental_stats
        stats["CacheMisses"] += 1

    def ensureLoadedAndLocked(self):
        if self._stats:
            return
        self._objectCache.lock.acquire()
        self._stats = PersistentJSONDict(os.path.join(self._objectCache.cacheDirectory(),
                                                      "stats.txt"))
        for k in ["CallsWithoutSourceFile",
                  "CallsWithMultipleSourceFiles",
                  "CallsWithPch",
                  "CallsForLinking",
                  "CacheEntries", "CacheSize",
                  "CacheHits", "CacheMisses",
                  "EvictedMisses", "HeaderChangedMisses",
                  "SourceChangedMisses", "ManifestsCount"]:
            if not k in self._stats:
                self._stats[k] = 0
        for key, value in self._incremental_stats.items():
            self._stats[key] += value
        self._incremental_stats = defaultdict(int)

    def resetCounters(self):
        self.ensureLoadedAndLocked()
        for k in ["CallsWithoutSourceFile",
                  "CallsWithMultipleSourceFiles",
                  "CallsWithPch",
                  "CallsForLinking",
                  "CacheHits", "CacheMisses",
                  "EvictedMisses", "HeaderChangedMisses",
                  "SourceChangedMisses"]:
            self._stats[k] = 0

    def save(self):
        self.ensureLoadedAndLocked()
        self._stats.save()
        self._objectCache.lock.release()
        self._stats = None  # Force reload stats when we'll re-acuire lock

class AnalysisResult:
    Ok, NoSourceFile, MultipleSourceFilesSimple, \
        MultipleSourceFilesComplex, CalledForLink, \
        CalledWithPch, ExternalDebugInfo = range(7)

def getFileHash(filePath, additionalData = None):
    hasher = HASH_ALGORITHM()
    with open(filePath, 'rb') as inFile:
        hasher.update(inFile.read())
    if additionalData is not None:
        hasher.update(additionalData)
    return hasher.hexdigest()

def getRelFileHash(filePath, baseDir):
    absFilePath = filePath
    if absFilePath.startswith(BASEDIR_REPLACEMENT):
        if not baseDir:
            raise LogicException('No CLCACHE_BASEDIR set, but found relative path ' + filePath)
        absFilePath = absFilePath.replace(BASEDIR_REPLACEMENT, baseDir, 1)
    if not os.path.exists(absFilePath):
        return None
    key = get_header_key(filePath, os.path.getmtime(absFilePath))
    result = HEADER_HASH_CACHE.get(key)
    if result is not None:
        return result
    result = getFileHash(absFilePath)
    HEADER_HASH_CACHE[key] = result
    return result

def getHash(data):
    hasher = HASH_ALGORITHM()
    hasher.update(data)
    return hasher.hexdigest()

def copyOrLink(srcFilePath, dstFilePath):
    if "CLCACHE_HARDLINK" in os.environ:
        ret = windll.kernel32.CreateHardLinkW(unicode(dstFilePath), unicode(srcFilePath), None)
        if ret != 0:
            # Touch the time stamp of the new link so that the build system
            # doesn't confused by a potentially old time on the file. The
            # hard link gets the same timestamp as the cached file.
            # Note that touching the time stamp of the link also touches
            # the time stamp on the cache (and hence on all over hard
            # links). This shouldn't be a problem though.
            os.utime(dstFilePath, None)
            return

    # If hardlinking fails for some reason (or it's not enabled), just
    # fall back to moving bytes around...
    copyfile(srcFilePath, dstFilePath)

def findCompilerBinary(pathVariable, hint):
    if hint:
        if hint in VERIFIED_COMPILER_HINTS:
            return hint
        if os.path.isfile(hint):
            VERIFIED_COMPILER_HINTS.add(hint)
            return hint
    compiler = COMPILER_PATH_CACHE.get(pathVariable)
    if compiler:
        return compiler
    compiler = findCompilerBinaryImpl(pathVariable)
    if compiler:
        COMPILER_PATH_CACHE[pathVariable] = compiler
    return compiler

def findCompilerBinaryImpl(pathVariable):
    if "CLCACHE_CL" in os.environ:
        path = os.environ["CLCACHE_CL"]
        return path if os.path.exists(path) else None

    frozenByPy2Exe = hasattr(sys, "frozen")
    if frozenByPy2Exe:
        myExecutablePath = unicode(sys.executable, sys.getfilesystemencoding()).upper()

    for dir in pathVariable.split(os.pathsep):
        path = os.path.join(dir, "cl.exe")
        if os.path.exists(path):
            if not frozenByPy2Exe:
                return path

            # Guard against recursively calling ourselves
            if path.upper() != myExecutablePath:
                return path
    return None


def printTraceStatement(msg):
    if "CLCACHE_LOG" in os.environ:
        script_dir = os.path.realpath(os.path.dirname(sys.argv[0]))
        print os.path.join(script_dir, "clcache.py") + " " + msg

def extractArgument(line, start, end):
    # If there are quotes from both sides of argument, remove them
    # "-Isome path" must becomse -Isome path
    if line[start] == '"' and line[end-1] == '"' and start != (end-1):
        start += 1
        end -= 1
    # Unescape quotes.
    return line[start:end].replace('\\"','"').strip()

def splitCommandsFile(line):
    # Note, we must treat lines in quotes as one argument. We do not use shlex
    # since seems it difficult to set up it to correctly parse escaped quotes.
    # A good test line to split is
    # '"-IC:\\Program files\\Some library" -DX=1 -DVERSION=\\"1.0\\"
    # -I..\\.. -I"..\\..\\lib" -DMYPATH=\\"C:\\Path\\"'
    i = 0
    wordStart = -1
    insideQuotes = False
    result = []
    while i < len(line):
        if line[i] == ' ' and not insideQuotes and wordStart >= 0:
            result.append(extractArgument(line, wordStart, i))
            wordStart = -1
        if line[i] == '"' and ((i == 0) or (i > 0 and line[i - 1] != '\\')):
            insideQuotes = not insideQuotes
        if line[i] != ' ' and wordStart < 0:
            wordStart = i
        i += 1

    if wordStart >= 0:
        result.append(extractArgument(line, wordStart, len(line)))
    return result

def expandCommandLine(cmdline):
    ret = []

    for arg in cmdline:
        if arg[0] == '@':
            includeFile = arg[1:]
            with open(includeFile, 'rb') as file:
                rawBytes = file.read()

            encoding = None

            encodingByBOM = {
                codecs.BOM_UTF32_BE: 'utf-32-be',
                codecs.BOM_UTF32_LE: 'utf-32-le',
                codecs.BOM_UTF16_BE: 'utf-16-be',
                codecs.BOM_UTF16_LE: 'utf-16-le',
            }

            for bom, enc in encodingByBOM.items():
                if rawBytes.startswith(bom):
                    encoding = encodingByBOM[bom]
                    rawBytes = rawBytes[len(bom):]
                    break

            includeFileContents = rawBytes.decode(encoding) if encoding is not None else rawBytes

            ret.extend(expandCommandLine(splitCommandsFile(includeFileContents.strip())))
        else:
            ret.append(arg)

    return ret

def parseCommandLine(cmdline):
    optionsWithParameter = ['Ob', 'Gs', 'Fa', 'Fd', 'Fm',
                            'Fp', 'FR', 'doc', 'FA', 'Fe',
                            'Fo', 'Fr', 'AI', 'FI', 'FU',
                            'D', 'U', 'I', 'Zp', 'vm',
                            'MP', 'Tc', 'V', 'wd', 'wo',
                            'W', 'Yc', 'Yl', 'Tp', 'we',
                            'Yu', 'Zm', 'F', 'Fi']
    options = defaultdict(list)
    responseFile = ""
    sourceFiles = []
    i = 0
    while i < len(cmdline):
        arg = cmdline[i]

        # Plain arguments startign with / or -
        if arg[0] == '/' or arg[0] == '-':
            isParametrized = False
            for opt in optionsWithParameter:
                if arg[1:len(opt)+1] == opt:
                    isParametrized = True
                    key = opt
                    if len(arg) > len(opt) + 1:
                        value = arg[len(opt)+1:]
                    else:
                        value = cmdline[i+1]
                        i += 1
                    options[key].append(value)
                    break

            if not isParametrized:
                options[arg[1:]] = []

        # Reponse file
        elif arg[0] == '@':
            responseFile = arg[1:]

        # Source file arguments
        else:
            sourceFiles.append(arg)

        i += 1

    return options, responseFile, sourceFiles

def analyzeCommandLine(cmdline):
    options, responseFile, sourceFiles = parseCommandLine(cmdline)
    compl = False

    # Technically, it would be possible to support /Zi: we'd just need to
    # copy the generated .pdb files into/out of the cache.
    if 'Zi' in options:
        return AnalysisResult.ExternalDebugInfo, None, None
    if 'Yu' in options:
        return AnalysisResult.CalledWithPch, None, None
    if 'Tp' in options:
        sourceFiles += options['Tp']
        compl = True
    if 'Tc' in options:
        sourceFiles += options['Tc']
        compl = True

    preprocessing = False

    for opt in ['E', 'EP', 'P']:
        if opt in options:
            preprocessing = True
            break

    if 'link' in options or (not 'c' in options and not preprocessing):
        return AnalysisResult.CalledForLink, None, None

    if len(sourceFiles) == 0:
        return AnalysisResult.NoSourceFile, None, None

    if len(sourceFiles) > 1:
        if compl:
            return AnalysisResult.MultipleSourceFilesComplex, None, None
        return AnalysisResult.MultipleSourceFilesSimple, sourceFiles, None

    outputFile = None
    if 'Fo' in options:
        outputFile = options['Fo'][0]

        if os.path.isdir(outputFile):
            srcFileName = os.path.basename(sourceFiles[0])
            outputFile = os.path.join(outputFile,
                                      os.path.splitext(srcFileName)[0] + ".obj")
    elif preprocessing:
        if 'P' in options:
            # Prerpocess to file.
            if 'Fi' in options:
                outputFile = options['Fi'][0]
            else:
                srcFileName = os.path.basename(sourceFiles[0])
                outputFile = os.path.join(os.getcwd(),
                                          os.path.splitext(srcFileName)[0] + ".i")
        else:
            # Prerocess to stdout. Use empty string rather then None to ease
            # output to log.
            outputFile = ''
    else:
        srcFileName = os.path.basename(sourceFiles[0])
        outputFile = os.path.join(os.getcwd(),
                                  os.path.splitext(srcFileName)[0] + ".obj")

    # Strip quotes around file names; seems to happen with source files
    # with spaces in their names specified via a response file generated
    # by Visual Studio.
    if outputFile.startswith('"') and outputFile.endswith('"'):
        outputFile = outputFile[1:-1]

    printTraceStatement("Compiler output file: '%s'" % outputFile)
    return AnalysisResult.Ok, sourceFiles[0], outputFile


def invokeRealCompiler(compilerBinary, cmdLine, captureOutput=False):
    realCmdline = [compilerBinary] + cmdLine
    if not '/showIncludes' in realCmdline:
        realCmdline.append('/showIncludes')

    printTraceStatement("Invoking real compiler as '%s'" % ' '.join(realCmdline))

    returnCode = None
    stdout = ''
    stderr = ''
    if captureOutput:
        compilerProcess = Popen(realCmdline, universal_newlines=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = compilerProcess.communicate()
        returnCode = compilerProcess.returncode
    else:
        returnCode = subprocess.call(realCmdline, universal_newlines=True)

    printTraceStatement("Real compiler returned code %d" % returnCode)
    return returnCode, stdout, stderr

# Given a list of Popen objects, removes and returns
# a completed Popen object.
#
# FIXME: this is a bit inefficient, Python on Windows does not appear
# to provide any blocking "wait for any process to complete" out of the
# box.
def waitForAnyProcess(procs):
    out = [p for p in procs if p.poll() != None]
    if len(out) >= 1:
        out = out[0]
        procs.remove(out)
        return out

    # Damn, none finished yet.
    # Do a blocking wait for the first one.
    # This could waste time waiting for one process while others have
    # already finished :(
    out = procs.pop(0)
    out.wait()
    return out

# Returns the amount of jobs which should be run in parallel when
# invoked in batch mode.
#
# The '/MP' option determines this, which may be set in cmdLine or
# in the CL environment variable.
def jobCount(cmdLine):
    switches = []

    if 'CL' in os.environ:
        switches.extend(os.environ['CL'].split(' '))

    switches.extend(cmdLine)

    mp_switch = [switch for switch in switches if re.search(r'^/MP\d+$', switch) != None]
    if len(mp_switch) == 0:
        return 1

    # the last instance of /MP takes precedence
    mp_switch = mp_switch.pop()

    count = mp_switch[3:]
    if count != "":
        return int(count)

    # /MP, but no count specified; use CPU count
    try:
        return multiprocessing.cpu_count()
    except:
        # not expected to happen
        return 2

# Run commands, up to j concurrently.
# Aborts on first failure and returns the first non-zero exit code.
def runJobs(commands, j=1):
    running = []
    returncode = 0

    while len(commands):

        while len(running) > j:
            thiscode = waitForAnyProcess(running).returncode
            if thiscode != 0:
                return thiscode

        thiscmd = commands.pop(0)
        running.append(Popen(thiscmd))

    while len(running) > 0:
        thiscode = waitForAnyProcess(running).returncode
        if thiscode != 0:
            return thiscode

    return 0


# re-invoke clcache.py once per source file.
# Used when called via nmake 'batch mode'.
# Returns the first non-zero exit code encountered, or 0 if all jobs succeed.
def reinvokePerSourceFile(cmdLine, sourceFiles):

    printTraceStatement("Will reinvoke self for: [%s]" % '] ['.join(sourceFiles))
    commands = []
    for sourceFile in sourceFiles:
        # The child command consists of clcache.py ...
        newCmdLine = [sys.executable]
        if not hasattr(sys, "frozen"):
            newCmdLine.append(sys.argv[0])

        for arg in cmdLine:
            # and the current source file ...
            if arg == sourceFile:
                newCmdLine.append(arg)
            # and all other arguments which are not a source file
            elif not arg in sourceFiles:
                newCmdLine.append(arg)

        printTraceStatement("Child: [%s]" % '] ['.join(newCmdLine))
        commands.append(newCmdLine)

    # TODO: Provide compiler output
    return runJobs(commands, jobCount(cmdLine)), ''

def printStatistics():
    cache = ObjectCache()
    with cache.lock:
        stats = CacheStatistics(cache)
        cfg = Configuration(cache)
        print """clcache statistics:
  current cache dir        : %s
  cache size               : %d bytes
  maximum cache size       : %d bytes
  cache entries            : %d
  manifests count          : %d
  cache hits               : %d
  cache misses             : %d
  called for linking       : %d
  called w/o sources       : %d
  calls w/ multiple sources: %d
  calls w/ PCH             : %d
  evicted misses           : %d
  header changed misses    : %d
  source changed misses    : %d
  """ % (
           cache.cacheDirectory(),
           stats.currentCacheSize(),
           cfg.maximumCacheSize(),
           stats.numCacheEntries(),
           stats.numManifests(),
           stats.numCacheHits(),
           stats.numCacheMisses(),
           stats.numCallsForLinking(),
           stats.numCallsWithoutSourceFile(),
           stats.numCallsWithMultipleSourceFiles(),
           stats.numCallsWithPch(),
           stats.numEvictedMisses(),
           stats.numHeaderChangedMisses(),
           stats.numSourceChangedMisses())

def resetStatistics():
    cache = ObjectCache()
    with cache.lock:
        stats = CacheStatistics(cache)
        stats.resetCounters()
        stats.save()
    print 'Statistics reset'

# Returns pair - list of includes and new compiler output.
# Output changes if strip is True in that case all lines with include
# directives are stripped from it
def getIncludes(compilerOutput, sourceFile, baseDir, strip):
    newOutput = []
    includesSet = set([])
    reFilePath = re.compile('^Note: including file: *(?P<file_path>.+)$')
    absSourceFile = os.path.normcase(os.path.abspath(sourceFile))
    if baseDir:
        baseDir = os.path.normcase(baseDir)
    for line in compilerOutput.split('\n'):
        match = reFilePath.match(line.rstrip('\r\n'))
        if match is not None:
            filePath = match.group('file_path')
            filePath = os.path.normcase(os.path.abspath(filePath))
            if filePath != absSourceFile:
                if baseDir and filePath.startswith(baseDir):
                    filePath = filePath.replace(baseDir, BASEDIR_REPLACEMENT, 1)
                includesSet.add(filePath)
        elif strip:
            newOutput.append(line)
    if strip:
        return list(includesSet), '\n'.join(newOutput)
    else:
        return list(includesSet), compilerOutput

def addObjectToCache(stats, cache, outputFile, compilerOutput, compilerStderr, cachekey, manifestHash):
    printTraceStatement("Adding file " + outputFile + " to cache using " +
                        "key " + cachekey)
    cache.setEntry(cachekey, outputFile, compilerOutput, compilerStderr, manifestHash)
    if outputFile != '':
        stats.registerCacheEntry(os.path.getsize(outputFile))
        cfg = Configuration(cache)
        cache.clean(stats, cfg.maximumCacheSize())

def processCacheHit(stats, cache, outputFile, cachekey):
    stats.registerCacheHit()
    stats.save()
    printTraceStatement("Reusing cached object for key " + cachekey + " for " +
                        "output file " + outputFile)
    if outputFile != '':
        if os.path.exists(outputFile):
            os.remove(outputFile)
        copyOrLink(cache.cachedObjectName(cachekey), outputFile)
    compilerOutput = cache.cachedCompilerOutput(cachekey)
    compilerStderr = cache.cachedCompilerStderr(cachekey)
    printTraceStatement("Finished. Exit code 0")
    return 0, compilerOutput, compilerStderr

def processObjectEvicted(stats, cache, outputFile, cachekey, compiler, cmdLine, manifestHash):
    stats.registerEvictedMiss()
    printTraceStatement("Cached object already evicted for key " + cachekey + " for " +
                        "output file " + outputFile)
    returnCode, compilerOutput, compilerStderr = invokeRealCompiler(compiler, cmdLine, captureOutput=True)
    if returnCode == 0 and (outputFile == '' or os.path.exists(outputFile)):
       addObjectToCache(stats, cache, outputFile, compilerOutput, compilerStderr, cacheke, manifestHash)
    stats.save()
    printTraceStatement("Finished. Exit code %d" % returnCode)
    return returnCode, compilerOutput, compilerStderr

def processHeaderChangedMiss(stats, cache, outputFile, manifest, manifestHash, keyInManifest, compiler, cmdLine):
    cachekey = cache.getDirectCacheKey(manifestHash, keyInManifest)
    stats.registerHeaderChangedMiss()
    returnCode, compilerOutput, compilerStderr = invokeRealCompiler(compiler, cmdLine, captureOutput=True)
    if returnCode == 0 and (outputFile == '' or os.path.exists(outputFile)):
        addObjectToCache(stats, cache, outputFile, compilerOutput, compilerStderr, cachekey, manifestHash)
        removedItems = []
        while len(manifest.hashes) >= MAX_MANIFEST_HASHES:
            key, objectHash = manifest.hashes.popitem()
            removedItems.append(objectHash)
        cache.removeObjects(stats, removedItems)
        manifest.hashes[keyInManifest] = cachekey
        if cache.setManifest(manifestHash, manifest):
            stats.registerManifest()
    stats.save()
    printTraceStatement("Finished. Exit code %d" % returnCode)
    return returnCode, compilerOutput, compilerStderr

def processNoManifestMiss(stats, cache, outputFile, manifestHash, baseDir, compiler, cmdLine, sourceFile):
    stats.registerSourceChangedMiss()
    stripIncludes = not '/showIncludes' in cmdLine
    returnCode, compilerOutput, compilerStderr = invokeRealCompiler(compiler, cmdLine, captureOutput=True)
    grabStderr = False
    # If these options present, cl.exe will list includes on stderr, not stdout
    for option in ['/E', '/EP', '/P']:
        if option in cmdLine:
            grabStderr = True
            break
    if grabStderr:
        listOfIncludes, compilerStderr = getIncludes(compilerStderr, sourceFile, baseDir, stripIncludes)
    else:
        listOfIncludes, compilerOutput = getIncludes(compilerOutput, sourceFile, baseDir, stripIncludes)
    manifest = Manifest(listOfIncludes, {})
    listOfHeaderHashes = [getRelFileHash(fileName, baseDir) for fileName in listOfIncludes]
    keyInManifest = cache.getKeyInManifest(listOfHeaderHashes)
    cachekey = cache.getDirectCacheKey(manifestHash, keyInManifest)

    if returnCode == 0 and (outputFile == '' or os.path.exists(outputFile)):
        addObjectToCache(stats, cache, outputFile, compilerOutput, compilerStderr, cachekey, manifestHash)
        manifest.hashes[keyInManifest] = cachekey
        if cache.setManifest(manifestHash, manifest):
            stats.registerManifest()
    stats.save()
    printTraceStatement("Finished. Exit code %d" % returnCode)
    return returnCode, compilerOutput, compilerStderr

def get_pipe_name(cacheDir):
    return '\\\\.\\pipe\\' + cacheDir.replace(':', '-').replace('\\','-')

def serveClient(hPipe):
    buffer = wintypes.create_string_buffer(PIPE_BUFFER_SIZE)

    ERROR_MORE_DATA = 234
    messages = []
    for i in range(4):
        inputMessage = ''
        while True:
            bytes_read = wintypes.c_int(0)
            result = windll.kernel32.ReadFile(hPipe,
                buffer,
                PIPE_BUFFER_SIZE,
                wintypes.addressof(bytes_read),
                wintypes.c_void_p(0))
            error = windll.kernel32.GetLastError()
            if not result and error != ERROR_MORE_DATA:
                print('Failed read pipe. Error {error}.'.
                       format(error=error))
                return 1
            inputMessage += buffer.raw[:bytes_read.value]
            if result:
                break
        # Decode to python string
        messages.append(unicode(inputMessage,'UTF-16'))
    pathVariable, includeVariable, currentDirectory, commandLine = messages
    os.chdir(currentDirectory)
    # Change env. since in other case child cl.exe may not launch since it will
    # fail to locate DLLs.
    # TODO: it is better to use it as argument of Popen instead.
    os.environ['PATH'] = pathVariable
    os.environ['INCLUDE'] = includeVariable
    expandedCommandLine = splitCommandsFile(commandLine)
    compilerHint = None
    if expandedCommandLine[1] == '/COMPILER:':
        compilerHint = expandedCommandLine[2]
        del expandedCommandLine[1:3]
    compiler = findCompilerBinary(pathVariable, compilerHint)
    if not compiler:
        print "Failed to locate cl.exe on PATH (and CLCACHE_CL is not set), aborting."
        exitCode = 1
        stdoutData = ''
    else:
        exitCode, stdoutData, stderrData = processCompileRequest(compiler, expandedCommandLine)
    response = struct.pack('@III', exitCode, len(stdoutData), len(stderrData))
    response += stdoutData
    response += stderrData
    responseBuffer = wintypes.create_string_buffer(response, len(response))
    result = windll.kernel32.WriteFile(hPipe,
            responseBuffer,
            len(responseBuffer),
            wintypes.addressof(bytes_read),
            wintypes.c_void_p(0))
    error = windll.kernel32.GetLastError()
    if not result:
        print('Failed write pipe. Error {error}.'.
                   format(error=error))
    windll.kernel32.FlushFileBuffers(hPipe)

def cacodaemonMain(daemonNumber):
    cache = ObjectCache()
    myDir = cache.getDaemonDir(os.getpid())
    sys.stdout = open(os.path.join(myDir, 'stdout.txt'), 'w')
    sys.stderr = open(os.path.join(myDir, 'stderr.txt'), 'w')
    pipeName = get_pipe_name(cache.cacheDirectory())
    PIPE_ACCESS_DUPLEX = 0x00000003
    PIPE_TYPE_MESSAGE = 0x00000004
    PIPE_READMODE_MESSAGE = 0x00000002
    PIPE_WAIT = 0x00000000
    PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
    PIPE_UNLIMITED_INSTANCES = 255
    INVALID_HANDLE_VALUE = 0xFFFFFFFF
    hPipe = windll.kernel32.CreateNamedPipeW(wintypes.c_wchar_p(pipeName),
        wintypes.c_int(PIPE_ACCESS_DUPLEX),
        wintypes.c_int(PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS),
        wintypes.c_int(PIPE_UNLIMITED_INSTANCES),
        wintypes.c_int(PIPE_BUFFER_SIZE),
        wintypes.c_int(PIPE_BUFFER_SIZE),
        wintypes.c_int(0),
        wintypes.c_void_p(0))
    if hPipe == INVALID_HANDLE_VALUE:
        print('Failed create pipe {name}. Error {error}.'.
                format(name=pipeName, error=windll.kernel32.GetLastError()))
        return 1
    while True:
        if not windll.kernel32.ConnectNamedPipe(hPipe, 0):
            print('Failed connect pipe {name}. Error {error}.'.
                    format(name=pipeName, error=windll.kernel32.GetLastError()))
            return 1
        serveClient(hPipe)
        if not windll.kernel32.DisconnectNamedPipe(hPipe):
            print('Failed disconnect pipe {name}. Error {error}.'.
                    format(name=pipeName, error=windll.kernel32.GetLastError()))
            return 1

def getDefaultDaemonCount():
    # The same algorithm, which used in ninja int GuessParallelism().
    cpuCount = multiprocessing.cpu_count()
    if cpuCount < 2:
        return cpuCount
    elif cpuCount == 2:
        return 3
    else:
        return cpuCount + 2

def spawnCacodaemons(count = 0):
    if "CLCACHE_DISABLE" in os.environ:
        print 'CLCACHE_DISABLE present in environment - do not spawn daemons'
        return 1
    if count == 0:
        count = getDefaultDaemonCount()
    # Kill already existing daemons, if any
    killCacodaemons()
    cache = ObjectCache()
    for i in range(count):
        args = [sys.executable, sys.argv[0], '--cacodaemon', str(i)]
        process = subprocess.Popen(args)
        cache.regiterDaemon(process.pid)
        print 'Spawned {pid}'.format(pid=process.pid)
    return 0

def killCacodaemons():
    cache = ObjectCache()
    pids = cache.getAllDaemonPids()
    PROCESS_TERMINATE = 0x01
    SYNCHRONIZE = 0x00100000
    INFINITE = 0xFFFFFFFF
    for pid in pids:
        print('Terminating {pid}..'.format(pid=pid))
        hProcess = windll.kernel32.OpenProcess(wintypes.c_int(PROCESS_TERMINATE | SYNCHRONIZE),
                                               wintypes.c_int(0),
                                               wintypes.c_int(pid))
        if not hProcess:
            print('Failed open process, error {error}'.
                format(error=windll.kernel32.GetLastError()))
            cache.unregiterDaemon(pid)
            continue
        if not windll.kernel32.TerminateProcess(hProcess, 1):
            print('Failed terminate process, error {error}'.
                format(error=windll.kernel32.GetLastError()))

        result = windll.kernel32.WaitForSingleObject(
            hProcess, wintypes.c_int(INFINITE))
        cache.unregiterDaemon(pid)
        windll.kernel32.CloseHandle(hProcess)

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print """\
    clcache.py v3.0.1
      --help   : show this help
      -s       : print cache statistics
      -z       : reset cache statistics
      -M <size>: set maximum cache size (in bytes)
      --spawn-cacodaemons <count> : spawns count cacodemon processes.
      --cacodaemon [<number>]  : used internally by spawn-cacodaemons.
      --kill-cacodaemons     : kill all cacodaemons associated with current cache dir.
    """
        return 0

    if len(sys.argv) == 2 and sys.argv[1] == "-s":
        printStatistics()
        return 0

    if len(sys.argv) == 2 and sys.argv[1] == "-z":
        resetStatistics()
        return 0

    if len(sys.argv) == 3 and sys.argv[1] == "-M":
        cache = ObjectCache()
        with cache.lock:
            cfg = Configuration(cache)
            cfg.setMaximumCacheSize(int(sys.argv[2]))
            cfg.save()
        return 0

    if len(sys.argv) >= 2 and sys.argv[1] == "--spawn-cacodaemons":
        daemonsCount = 0
        if len(sys.argv) >= 3:
            daemonsCount = int(sys.argv[2])
        return spawnCacodaemons(daemonsCount)

    if len(sys.argv) == 3 and sys.argv[1] == "--cacodaemon":
        cacodaemonMain(int(sys.argv[2]))
        return 0

    if len(sys.argv) == 2 and sys.argv[1] == "--kill-cacodaemons":
        killCacodaemons()
        return 0
    compiler = findCompilerBinary(os.environ["PATH"], None)
    if not compiler:
        print "Failed to locate cl.exe on PATH (and CLCACHE_CL is not set), aborting."
        return 1

    printTraceStatement("Found real compiler binary at '%s'" % compiler)

    if "CLCACHE_DISABLE" in os.environ:
        return invokeRealCompiler(compiler, sys.argv[1:])[0]
    try:
        exitCode, compilerStdout, compilerStderr = processCompileRequest(compiler, sys.argv)
        sys.stdout.write(compilerStdout)
        sys.stderr.write(compilerStderr)
        return exitCode
    except LogicException as e:
        print e
        return 1

def processCompileRequest(compiler, args):
    printTraceStatement("Parsing given commandline '%s'" % args[1:] )

    cmdLine = expandCommandLine(args[1:])
    printTraceStatement("Expanded commandline '%s'" % cmdLine )
    analysisResult, sourceFile, outputFile = analyzeCommandLine(cmdLine)

    if analysisResult == AnalysisResult.MultipleSourceFilesSimple:
        return reinvokePerSourceFile(cmdLine, sourceFile), '', ''

    cache = ObjectCache()
    stats = CacheStatistics(cache)
    if analysisResult != AnalysisResult.Ok:
        with cache.lock:
            if analysisResult == AnalysisResult.NoSourceFile:
                printTraceStatement("Cannot cache invocation as %s: no source file found" % (' '.join(cmdLine)) )
                stats.registerCallWithoutSourceFile()
            elif analysisResult == AnalysisResult.MultipleSourceFilesComplex:
                printTraceStatement("Cannot cache invocation as %s: multiple source files found" % (' '.join(cmdLine)) )
                stats.registerCallWithMultipleSourceFiles()
            elif analysisResult == AnalysisResult.CalledWithPch:
                printTraceStatement("Cannot cache invocation as %s: precompiled headers in use" % (' '.join(cmdLine)) )
                stats.registerCallWithPch()
            elif analysisResult == AnalysisResult.CalledForLink:
                printTraceStatement("Cannot cache invocation as %s: called for linking" % (' '.join(cmdLine)) )
                stats.registerCallForLinking()
            elif analysisResult == AnalysisResult.ExternalDebugInfo:
                printTraceStatement("Cannot cache invocation as %s: external debug information (/Zi) is not supported" % (' '.join(cmdLine)) )
            stats.save()
        return invokeRealCompiler(compiler, args[1:])
    if 'CLCACHE_NODIRECT' in os.environ:
        return processNoDirect(stats, cache, compiler, cmdLine)
    manifestHash = cache.getManifestHash(compiler, cmdLine, sourceFile)
    manifest = cache.getManifest(manifestHash)
    baseDir = os.environ.get('CLCACHE_BASEDIR')
    if baseDir and not baseDir.endswith(os.path.sep):
        baseDir += os.path.sep
    hasMissedHeader = False
    if manifest is not None:
        # NOTE: command line options already included in hash for manifest name
        listOfHeaderHashes = []
        for fileName in manifest.includeFiles:
            fileHash = getRelFileHash(fileName, baseDir)
            if fileHash is not None:
                # May be if source does not use this header anymore (e.g. if that
                # header was included through some other header, which now changed).
                listOfHeaderHashes.append(fileHash)
            else:
                hasMissedHeader = True
                break
        # If includes set changed, we MUST re-create manifest, on other case
        # following problem may appear:
        # 1. file.cpp uses #include "header.h" and compiled with -I dir -I dir\subdir
        # 2. First time header.h is found in dir\, so it saved to manifest
        # 3. Build directory cleared, header.h moved to dir\subdir
        # 4. Now we don't found dir\header.h, so run real compiler and cache
        #    result NOT USING header.h hash AT ALL
        # 5. Build directory clean again, header.h still in dir\subdir,
        #    but has changed it contents
        # 6. We still using manifest from step 2, we still do not use hash for
        #    header.h, so use cached result from step 4,
        #    which is incorrect.
        if hasMissedHeader:
            return processNoManifestMiss(stats, cache, outputFile, manifestHash, baseDir, compiler, cmdLine, sourceFile)
        keyInManifest = cache.getKeyInManifest(listOfHeaderHashes)
        cachekey = manifest.hashes.get(keyInManifest)
        if cachekey is not None:
            if cache.hasEntry(cachekey):
                return processCacheHit(stats, cache, outputFile, cachekey)
            else:
                return processObjectEvicted(stats, cache, outputFile, cachekey, compiler, cmdLine, manifestHash)
        else:
            # Some of header files changed - recompile and add to manifest
            return processHeaderChangedMiss(stats, cache, outputFile, manifest, manifestHash, keyInManifest, compiler, cmdLine)
    else:
        return processNoManifestMiss(stats, cache, outputFile, manifestHash, baseDir, compiler, cmdLine, sourceFile)

def processNoDirect(stats, cache, compiler, cmdLine):
    cachekey = cache.computeKey(compiler, cmdLine)
    if cache.hasEntry(cachekey):
        with cache.lock:
            stats.registerCacheHit()
            stats.save()
        printTraceStatement("Reusing cached object for key " + cachekey + " for " +
                            "output file " + outputFile)
        if os.path.exists(outputFile):
            os.remove(outputFile)
        copyOrLink(cache.cachedObjectName(cachekey), outputFile)
        compilerStdout = cache.cachedCompilerOutput(cachekey)
        compilerStderr = cache.cachedCompilerStderr(cachekey)
        printTraceStatement("Finished. Exit code 0")
        return 0, compilerStdout, compilerStderr
    else:
        returnCode, compilerStdout, compilerStderr = invokeRealCompiler(compiler, cmdLine, captureOutput=True)
        with cache.lock:
            stats.registerCacheMiss()
            if returnCode == 0 and os.path.exists(outputFile):
                printTraceStatement("Adding file " + outputFile + " to cache using " +
                                    "key " + cachekey)
                cache.setEntry(cachekey, outputFile, compilerStdout, compilerStderr, None)
                stats.registerCacheEntry(os.path.getsize(outputFile))
                cfg = Configuration(cache)
                cache.clean(stats, cfg.maximumCacheSize())
            stats.save()
        printTraceStatement("Finished. Exit code %d" % returnCode)
        return returnCode, compilerStdout, compilerStderr

if __name__ == '__main__':
    sys.exit(main())
