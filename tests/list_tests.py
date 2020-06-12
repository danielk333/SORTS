import subprocess

p = subprocess.Popen('pytest --collect-only -qq', cwd='.', shell=True, stdout=subprocess.PIPE)
out = p.communicate()[0]

fout = 'tests/test_list.txt'

with open(fout,'w') as f:
        f.write(out)
