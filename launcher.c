#include <Windows.h>
#include <stdio.h>

#define BUFFER_SIZE 4096

#pragma pack(push, 1)
typedef struct tagRESULT_BLOCK_HEADER{
  DWORD exit_code;
  DWORD stdout_length;
  DWORD stderr_length;
}RESULT_BLOCK_HEADER;
#pragma pack(pop)

/* Returns length of the string, excluding terminating null,
   or -1 if error occurs
 */
int generate_pipe_name(wchar_t* dst, const wchar_t* clcache_dir) {
  #define PIPE_PREFIX L"\\\\.\\pipe\\"
  wchar_t* p_d = dst + wcslen(PIPE_PREFIX);
  const wchar_t* p_s = clcache_dir;
  wcscpy(dst, PIPE_PREFIX);
  for (; *p_s; ++p_s) {
    if ((p_d - dst) >= BUFFER_SIZE)
      return 0;
    if (*p_s == L'\\' || *p_s == L':')
      *p_d++ = L'-';
    else
      *p_d++ = *p_s;
  }
  *p_d = *p_s;
  return 1;
}

int transfer_to_std_handle(HANDLE h_pipe, DWORD length, DWORD std_handle_id) {
  DWORD data_left = length;
  DWORD current_transfer_size = 0;
  static BYTE buffer[BUFFER_SIZE];
  DWORD bytes_transferred = 0;
  HANDLE h_stdout = GetStdHandle(std_handle_id);
  while (data_left > 0) {
    current_transfer_size = data_left > BUFFER_SIZE ? BUFFER_SIZE : data_left;
    if (!ReadFile(
        h_pipe,
        &buffer[0],
        current_transfer_size,
        &bytes_transferred,
        NULL)) {
      fprintf(stderr,"Unexpected error during reading from pipe %d.\n", GetLastError());
      return 0;
    }
    if (!WriteFile(
        h_stdout,
        &buffer,
        current_transfer_size,
        &bytes_transferred,
        NULL)) {
      fprintf(stderr,"Unexpected error during writing to pipe. %d\n", GetLastError());
      return 0;
    }
    data_left -= current_transfer_size;
  }
  return 1;
}

int send_wstring(HANDLE h_pipe, const wchar_t* str) {
  DWORD bytes_transferred = 0;
  if (!WriteFile(
      h_pipe,
      str,
      wcslen(str) * sizeof(wchar_t),
      &bytes_transferred,
      NULL)) {
    fprintf(stderr,"Unexpected error during writing to pipe. %d\n", GetLastError());
    return 0;
  }
  return 1;
}

// Warning: returns pointer to static buffer. It will be overwritten on next
// function call
const wchar_t* get_env_variable(const wchar_t* name) {
  static wchar_t env_buffer[BUFFER_SIZE];
  DWORD result = GetEnvironmentVariable(name,
                                  env_buffer,
                                  BUFFER_SIZE);
  if (result == 0 || result > BUFFER_SIZE)
    env_buffer[0] = 0;
  return env_buffer;
}

int try_do_work(const wchar_t* pipe_name, DWORD* exit_code) {
  DWORD bytes_transferred = 0;
  RESULT_BLOCK_HEADER header;
  static wchar_t current_directory[MAX_PATH + 1];
  HANDLE h_pipe = CreateFile(
         pipe_name,
         GENERIC_READ |  GENERIC_WRITE,
         0,              // no sharing
         NULL,           // default security attributes
         OPEN_EXISTING,  // opens existing pipe
         0,              // default attributes
         NULL);          // no template file
  if (h_pipe == INVALID_HANDLE_VALUE) {
    int last_error = GetLastError();
    if (last_error != ERROR_PIPE_BUSY)
      fprintf(stderr,"Unexpected error during opening pipe. %d\n", GetLastError());
    return 0;
  }
  if (!GetCurrentDirectory(
        sizeof(current_directory)/sizeof(current_directory[1]),
        current_directory)) {
    fprintf(stderr,"Failed get current directory. %d\n", GetLastError());
    return 0;
  }
  if (!send_wstring(h_pipe, get_env_variable(L"PATH")))
    return 0;
  if (!send_wstring(h_pipe, get_env_variable(L"INCLUDE")))
    return 0;
  if (!send_wstring(h_pipe, current_directory))
    return 0;
  if (!send_wstring(h_pipe,  GetCommandLineW())) {
    fprintf(stderr, "Failed send command line\n");
    return 0;
  }

  if (!ReadFile(
      h_pipe,
      &header,
      sizeof(header),
      &bytes_transferred,
      NULL)) {
    fprintf(stderr, "Unexpected error during reading from pipe %d.\n", GetLastError());
    return 0;
  }
  *exit_code = header.exit_code;
  if (!transfer_to_std_handle(h_pipe, header.stdout_length, STD_OUTPUT_HANDLE))
    return 0;
  if (!transfer_to_std_handle(h_pipe, header.stderr_length, STD_ERROR_HANDLE))
    return 0;
  return 1;
}

int main(int argc, char* argv[]) {
  static wchar_t pipe_name[BUFFER_SIZE];
  int i = 0;
  int exit_code = 0;
  const wchar_t* clcache_dir = NULL;

  clcache_dir = get_env_variable(L"CLCACHE_DIR");
  if (clcache_dir[0] == 0) {
    fprintf(stderr, "Failed get CLCACHE_DIR environment variable.\n");
    return 1;
  }
  if (!generate_pipe_name(pipe_name, clcache_dir)) {
    fprintf(stderr,"CLCACHE_DIR Too large.\n");
    return 1;
  }
  while(1) {
    if (try_do_work(pipe_name, &exit_code)) {
      return exit_code;
    }
    if (!WaitNamedPipe(pipe_name, NMPWAIT_WAIT_FOREVER)) {
      fprintf(stderr,"Failed wait for named pipe. Error %d\n", GetLastError());
      return -1;
    }
  }
  return 0;
}
