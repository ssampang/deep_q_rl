proc = subprocess.Popen(['gnugo','--quiet','--mode','gtp'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
proc.stdin.write('play black A3')
fcntl.fcntl(proc.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
proc.stdout.read()
