import os
modules='joblib Ipython sklearn seaborn flask webptools pysftp numpy requests'.split(' ')
fn='versions.txt'
os.system('pip freeze > '+fn)
installed=''
with open(fn) as f:
    installed += f.read()
    
for module in modules:
    if(module+'==' not in  installed):
        os.system('pip3 install '+module)
        
#help('modules')
#print(list(sys.modules))