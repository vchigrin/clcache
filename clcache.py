#!/usr/bin/env python
#
# clcache.py - a compiler cache for Microsoft Visual Studio
#
# Copyright (c) 2010, 2011, 2012 froglogic GmbH <raabe@froglogic.com>
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
import codecs
from collections import defaultdict, namedtuple
import cPickle as pickle

from filelock import FileLock
import hashlib
import json
import os
from shutil import copyfile, rmtree
import subprocess
from subprocess import Popen, PIPE, STDOUT
import sys
import tempfile
import multiprocessing
import re

Manifest = namedtuple('Manifest', ['includeFiles', 'hashes'])

# Manifest file will have at most this number of hash lists in it. Need to avoid
# manifests grow too large.
MAX_MANIFEST_HASHES = 100

def cacheLock(cache):
    lock = FileLock("x", timeout=100)
    lock.lockfile = os.path.join(cache.cacheDirectory(), "cache.lock")
    return lock

class ObjectCache:
    def __init__(self):
        try:
            self.dir = os.environ["CLCACHE_DIR"]
        except KeyError:
            self.dir = os.path.join(os.path.expanduser("~"), "clcache")
        self.tempDir = os.path.join(self.dir, '.temp')
        # Creates both self.dir and self.tempDir if neccessary
        if not os.path.exists(self.tempDir):
            os.makedirs(self.tempDir)

    def cacheDirectory(self):
        return self.dir

    def clean(self, stats, maximumSize):
        # Ensure other processes will not touch cache during cleaning
        stats.ensureLoadedAndLocked()
        currentSize = stats.currentCacheSize()
        if currentSize < maximumSize:
            return
        # Free at least 10% to avoid too often cleanups and performance degradation
        # on large caches.
        maxSizeAfterCleanup = maximumSize * 0.9

        objects = [os.path.join(root, "object")
                   for root, folder, files in os.walk(self.dir)
                   if "object" in files]

        objectInfos = [(os.stat(fn), fn) for fn in objects]
        objectInfos.sort(key=lambda t: t[0].st_atime, reverse=True)
        for stat, fn in objectInfos:
            rmtree(os.path.split(fn)[0])
            currentSize -= stat.st_size
            if currentSize < maxSizeAfterCleanup:
                break
        stats.setCacheSize(currentSize)

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

    def hasEntry(self, key):
        return os.path.exists(self.cachedObjectName(key))

    def setEntry(self, key, objectFileName, compilerOutput):
        if not os.path.exists(self._cacheEntryDir(key)):
            os.makedirs(self._cacheEntryDir(key))
        copyfile(objectFileName, self.cachedObjectName(key))
        open(self._cachedCompilerOutputName(key), 'w').write(compilerOutput)

    def setManifest(self, manifestHash, manifest):
        if not os.path.exists(self._cacheEntryDir(manifestHash)):
            os.makedirs(self._cacheEntryDir(manifestHash))
        with open(self._manifestName(manifestHash), 'wb') as outFile:
            pickle.dump(manifest, outFile)

    def getManifest(self, manifestHash):
        fileName = self._manifestName(manifestHash)
        if not os.path.exists(fileName):
            return None
        # TODO: Locking if somebody decide compile the same file with the same
        # options on the same cache (A bit paranoid situation)
        with open(fileName, 'rb') as inFile:
            return pickle.load(inFile)
        return manifest

    def cachedObjectName(self, key):
        return os.path.join(self._cacheEntryDir(key), "object")

    def cachedCompilerOutput(self, key):
        return open(self._cachedCompilerOutputName(key), 'r').read()

    def getTempFilePath(self, sourceFile):
        ext =  os.path.splitext(sourceFile)[1]
        handle, path = tempfile.mkstemp(suffix=ext, dir=self.tempDir)
        os.close(handle)
        return path

    def _cacheEntryDir(self, key):
        return os.path.join(self.dir, key[:2], key)

    def _cachedCompilerOutputName(self, key):
        return os.path.join(self._cacheEntryDir(key), "output.txt")

    def _manifestName(self, key):
        return os.path.join(self._cacheEntryDir(key), "manifest.dat")

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
        self._cfg.save()


class CacheStatistics:
    def __init__(self, objectCache, cacheLock):
        # Use two dictionaries to ensure we'll grab cache lock on the smallest
        # possible time. We collect increment _incremental_stats while possible
        # and then merge it with stats on disk.
        self._incremental_stats = defaultdict(int)
        self._stats = None
        self._cacheLock = cacheLock
        self._objectCache = objectCache

    def numCallsWithoutSourceFile(self):
        self.ensureLoadedAndLocked()
        return self._stats["CallsWithoutSourceFile"]

    def registerCallWithoutSourceFile(self):
        self._incremental_stats["CallsWithoutSourceFile"] += 1

    def numCallsWithMultipleSourceFiles(self):
        self.ensureLoadedAndLocked()
        return self._stats["CallsWithMultipleSourceFiles"]

    def registerCallWithMultipleSourceFiles(self):
        self._incremental_stats["CallsWithMultipleSourceFiles"] += 1

    def numCallsWithPch(self):
        self.ensureLoadedAndLocked()
        return self._stats["CallsWithPch"]

    def registerCallWithPch(self):
        self._incremental_stats["CallsWithPch"] += 1

    def numCallsForLinking(self):
        self.ensureLoadedAndLocked()
        return self._stats["CallsForLinking"]

    def registerCallForLinking(self):
        self._incremental_stats["CallsForLinking"] += 1

    def numEvictedMisses(self):
        self.ensureLoadedAndLocked()
        return self._stats["EvictedMisses"]

    def registerEvictedMiss(self):
        self._incremental_stats["EvictedMisses"] += 1

    def numHeaderChangedMisses(self):
        self.ensureLoadedAndLocked()
        return self._stats["HeaderChangedMisses"]

    def registerHeaderChangedMiss(self):
        self._incremental_stats["HeaderChangedMisses"] += 1

    def numSourceChangedMisses(self):
        self.ensureLoadedAndLocked()
        return self._stats["SourceChangedMisses"]

    def registerSourceChangedMiss(self):
        self._incremental_stats["SourceChangedMisses"] += 1

    def numCacheEntries(self):
        self.ensureLoadedAndLocked()
        return self._stats["CacheEntries"]

    def registerCacheEntry(self, size):
        self._incremental_stats["CacheEntries"] += 1
        self._incremental_stats["CacheSize"] += size

    def currentCacheSize(self):
        self.ensureLoadedAndLocked()
        return self._stats["CacheSize"]

    def setCacheSize(self, size):
        self.ensureLoadedAndLocked()
        self._stats["CacheSize"] = size

    def numCacheHits(self):
        self.ensureLoadedAndLocked()
        return self._stats["CacheHits"]

    def registerCacheHit(self):
        self._incremental_stats["CacheHits"] += 1

    def numCacheMisses(self):
        self.ensureLoadedAndLocked()
        return self._stats["CacheMisses"]

    def registerCacheMiss(self):
        self._incremental_stats["CacheMisses"] += 1

    def ensureLoadedAndLocked(self):
        if self._stats:
            return
        self._cacheLock.acquire()
        self._stats = PersistentJSONDict(os.path.join(self._objectCache.cacheDirectory(),
                                                      "stats.txt"))
        for k in ["CallsWithoutSourceFile",
                  "CallsWithMultipleSourceFiles",
                  "CallsWithPch",
                  "CallsForLinking",
                  "CacheEntries", "CacheSize",
                  "CacheHits", "CacheMisses",
                  "EvictedMisses", "HeaderChangedMisses",
                  "SourceChangedMisses"]:
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
        self._cacheLock.release()
        self._stats = None  # Force reload stats when we'll re-acuire lock



class AnalysisResult:
    Ok, NoSourceFile, MultipleSourceFilesSimple, \
        MultipleSourceFilesComplex, CalledForLink, \
        CalledWithPch = range(6)

def getFileHash(filePath, additionalData = None):
    sha = hashlib.sha1()
    with open(filePath, 'rb') as inFile:
        sha.update(inFile.read())
    if additionalData is not None:
        sha.update(additionalData)
    return sha.hexdigest()

def getHash(data):
    sha = hashlib.sha1()
    sha.update(data)
    return sha.hexdigest()

def findCompilerBinary():
    try:
        path = os.environ["CLCACHE_CL"]
        if os.path.exists(path):
            return path
    except KeyError:
        for dir in os.environ["PATH"].split(os.pathsep):
            path = os.path.join(dir, "cl.exe")
            if os.path.exists(path):
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
    return line[start:end].replace('\\"','"')

def splitCommandsFile(line):
    # Note, we must treat lines in quotes as one argument. We do not use shlex
    # since seems it difficult to set up it to correctly parse escaped quotes.
    # A good test line to split is
    # '"-IC:\\Program files\\Some library" -DX=1 -DVERSION=\\"1.0\\"
    # -I..\\.. -I"..\\..\\lib" -DMYPATH=\\"C:\\Path\\"'
    i = 0
    wordStart = -1
    inside_quotes = False
    result = []
    while i < len(line):
        if line[i] == ' ' and not inside_quotes and wordStart >= 0:
            result.append(extractArgument(line, wordStart, i))
            wordStart = -1
        if line[i] == '"' and ((i == 0) or (i > 0 and line[i - 1] != '\\')):
            inside_quotes = not inside_quotes
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

            ret.extend(expandCommandLine(splitCommandsFile(includeFileContents)))
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
                            'Yu', 'Zm', 'F']
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

    if 'Yu' in options:
        return AnalysisResult.CalledWithPch, None, None
    if 'Tp' in options:
        sourceFiles += options['Tp']
        compl = True
    if 'Tc' in options:
        sourceFiles += options['Tc']
        compl = True

    if 'link' in options or not 'c' in options:
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
    else:
        srcFileName = os.path.basename(sourceFiles[0])
        outputFile = os.path.join(os.getcwd(),
                                  os.path.splitext(srcFileName)[0] + ".obj")

    if os.path.isdir(outputFile):
        srcFileName = os.path.basename(sourceFiles[0])
        outputFile = os.path.join(outputFile,
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
    printTraceStatement("Invoking real compiler as '%s'" % ' '.join(realCmdline))

    returnCode = None
    output = None
    if captureOutput:
        compilerProcess = Popen(realCmdline, stdout=PIPE, stderr=STDOUT)
        output = compilerProcess.communicate()[0].replace('\r\n','\n')
        returnCode = compilerProcess.returncode
    else:
        returnCode = subprocess.call(realCmdline)

    printTraceStatement("Real compiler returned code %d" % returnCode)
    return returnCode, output

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
        newCmdLine = [sys.executable, sys.argv[0]]

        for arg in cmdLine:
            # and the current source file ...
            if arg == sourceFile:
                newCmdLine.append(arg)
            # and all other arguments which are not a source file
            elif not arg in sourceFiles:
                newCmdLine.append(arg)

        printTraceStatement("Child: [%s]" % '] ['.join(newCmdLine))
        commands.append(newCmdLine)

    return runJobs(commands, jobCount(cmdLine))

def printStatistics():
    cache = ObjectCache()
    stats = CacheStatistics(cache, cacheLock(cache))
    cfg = Configuration(cache)
    print """clcache statistics:
  current cache dir        : %s
  cache size               : %d bytes
  maximum cache size       : %d bytes
  cache entries            : %d
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
  stats = CacheStatistics(cache, cacheLock(cache))
  stats.resetCounters()
  stats.save()
  print 'Statistics reset'

def parsePreprocessorOutput(preprocessorOutput, sourceFile):
    includesSet = set([])
    reFilePath = re.compile('^#line \\d+ \\"?(?P<file_path>[^\\"]+)\\"?$')
    absSourceFile = os.path.abspath(sourceFile)
    # TODO: Add support for CCACHE_BASEDIR to allow cache hits when repo dir is renamed.
    for line in preprocessorOutput:
        match = reFilePath.match(line.rstrip('\r\n'))
        if match is not None:
            filePath = match.group('file_path').replace('\\\\', '\\')
            if filePath != absSourceFile:
                includesSet.add(filePath)
    return list(includesSet)

def preprocessFile(compilerBinary, commandLine, sourceFile, cache, captureOutput=False):
    preprocessedFilePath = None
    if captureOutput:
        preprocessedFilePath = cache.getTempFilePath(sourceFile)
        ppcmd = [compilerBinary, "/P", "/Fi" + preprocessedFilePath]
    else:
        ppcmd = [compilerBinary, "/E"]
    ppcmd += [arg for arg in commandLine[1:] if not arg in ("-c", "/c")]
    preprocessor = Popen(ppcmd, stdout=PIPE, stderr=PIPE)
    (ppout, pperr) = preprocessor.communicate()

    if preprocessor.returncode != 0:
        sys.stderr.write(pperr)
        sys.stderr.write("clcache: preprocessor failed\n")
        sys.exit(preprocessor.returncode)
    if captureOutput:
        with open(preprocessedFilePath, 'r') as inFile:
            listOfIncludes = parsePreprocessorOutput(inFile, sourceFile)
    else:
        listOfIncludes = parsePreprocessorOutput(ppout.split('\n'), sourceFile)
    return listOfIncludes, preprocessedFilePath

def addObjectToCache(cache, outputFile, compilerOutput, cachekey):
    printTraceStatement("Adding file " + outputFile + " to cache using " +
                        "key " + cachekey)
    cache.setEntry(cachekey, outputFile, compilerOutput)
    stats.registerCacheEntry(os.path.getsize(outputFile))
    cfg = Configuration(cache)
    cache.clean(stats, cfg.maximumCacheSize())

def processCacheHit(stats, cache, outputFile, cachekey):
    stats.registerCacheHit()
    stats.save()
    printTraceStatement("Reusing cached object for key " + cachekey + " for " +
                        "output file " + outputFile)
    copyfile(cache.cachedObjectName(cachekey), outputFile)
    sys.stdout.write(cache.cachedCompilerOutput(cachekey))
    printTraceStatement("Finished. Exit code 0")
    sys.exit(0)

def processObjectEvicted(stats, cache, outputFile, cachekey, compiler, cmdLine):
    stats.registerEvictedMiss()
    printTraceStatement("Cached object already evicted for key " + cachekey + " for " +
                        "output file " + outputFile)
    returnCode, compilerOutput = invokeRealCompiler(compiler, cmdLine, captureOutput=True)
    if returnCode == 0 and os.path.exists(outputFile):
       addObjectToCache(cache, outputFile, compilerOutput, cachekey)
    stats.save()
    sys.stdout.write(compilerOutput)
    printTraceStatement("Finished. Exit code %d" % returnCode)
    sys.exit(returnCode)

def processHeaderChangedMiss(stats, cache, outputFile, manifest, manifestHash, includesKey, compiler, cmdLine):
    cachekey = getHash(includesKey)
    stats.registerHeaderChangedMiss()
    returnCode, compilerOutput = invokeRealCompiler(compiler, cmdLine, captureOutput=True)
    if returnCode == 0 and os.path.exists(outputFile):
        addObjectToCache(cache, outputFile, compilerOutput, cachekey)
        while len(manifest.hashes) >= MAX_MANIFEST_HASHES:
            manifest.hashes.popitem()
        manifest.hashes[includesKey] = cachekey
        cache.setManifest(manifestHash, manifest)
    stats.save()
    sys.stdout.write(compilerOutput)
    printTraceStatement("Finished. Exit code %d" % returnCode)
    sys.exit(returnCode)

def processNoManifestMiss(stats, cache, outputFile, manifestHash, compiler, cmdLine):
    stats.registerSourceChangedMiss()
    singlePreprocess = not 'CLCACHE_CPP2' in os.environ
    listOfIncludes, preprocessedFile = preprocessFile(compiler, cmdLine, sourceFile, cache, singlePreprocess)
    manifest = Manifest(listOfIncludes, {})
    listOfHashes = [getFileHash(fileName) for fileName in listOfIncludes]
    includesKey = getHash(','.join(listOfHashes))
    if singlePreprocess:
        index = cmdLine.index(sourceFile)
        cmdLine[index] = preprocessedFile
    cachekey = getHash(includesKey)
    returnCode, compilerOutput = invokeRealCompiler(compiler, cmdLine, captureOutput=True)
    if singlePreprocess:
        os.remove(preprocessedFile)
    if returnCode == 0 and os.path.exists(outputFile):
        addObjectToCache(cache, outputFile, compilerOutput, cachekey)
        manifest.hashes[includesKey] = cachekey
        cache.setManifest(manifestHash, manifest)
    stats.save()
    sys.stdout.write(compilerOutput)
    printTraceStatement("Finished. Exit code %d" % returnCode)
    sys.exit(returnCode)

if len(sys.argv) == 2 and sys.argv[1] == "--help":
    print """\
clcache.py v0.1"
  --help   : show this help
  -s       : print cache statistics
  -z       : reset cache statistics
  -M <size>: set maximum cache size (in bytes)
"""
    sys.exit(0)

if len(sys.argv) == 2 and sys.argv[1] == "-s":
    printStatistics()
    sys.exit(0)

if len(sys.argv) == 2 and sys.argv[1] == "-z":
    resetStatistics()
    sys.exit(0)

if len(sys.argv) == 3 and sys.argv[1] == "-M":
    cache = ObjectCache()
    cfg = Configuration(cache)
    cfg.setMaximumCacheSize(int(sys.argv[2]))
    cfg.save()
    sys.exit(0)

compiler = findCompilerBinary()
if not compiler:
    print "Failed to locate cl.exe on PATH (and CLCACHE_CL is not set), aborting."
    sys.exit(1)

printTraceStatement("Found real compiler binary at '%s'" % compiler)

if "CLCACHE_DISABLE" in os.environ:
    sys.exit(invokeRealCompiler(compiler, sys.argv[1:])[0])

printTraceStatement("Parsing given commandline '%s'" % sys.argv[1:] )

cmdLine = expandCommandLine(sys.argv[1:])
printTraceStatement("Expanded commandline '%s'" % cmdLine )
analysisResult, sourceFile, outputFile = analyzeCommandLine(cmdLine)

if analysisResult == AnalysisResult.MultipleSourceFilesSimple:
    sys.exit(reinvokePerSourceFile(cmdLine, sourceFile))

cache = ObjectCache()
lock = cacheLock(cache)
stats = CacheStatistics(cache, lock)

if analysisResult != AnalysisResult.Ok:
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
    stats.save()
    sys.exit(invokeRealCompiler(compiler, sys.argv[1:])[0])

manifestHash = cache.getManifestHash(compiler, cmdLine, sourceFile)
manifest = cache.getManifest(manifestHash)
if manifest is not None:
    # NOTE: command line options already included in hash for manifest name
    listOfHashes = [getFileHash(fileName) for fileName in manifest.includeFiles]
    includesKey = getHash(','.join(listOfHashes))
    cachekey = manifest.hashes.get(includesKey)
    if cachekey is not None:
        if cache.hasEntry(cachekey):
            processCacheHit(stats, cache, outputFile, cachekey)
        else:
            processObjectEvicted(stats, cache, outputFile, cachekey, compiler, cmdLine)
    else:
        # Some of header files changed - recompile and add to manifest
        processHeaderChangedMiss(stats, cache, outputFile, manifest, manifestHash, includesKey, compiler, cmdLine)
else:
    processNoManifestMiss(stats, cache, outputFile, manifestHash, compiler, cmdLine)
