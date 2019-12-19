import os
from shutil import copyfile

if not os.path.exists("/content/drive/My Drive/ashrae"):
    from google.colab import drive
    drive.mount('/content/drive')

if os.path.exists("./ashrae"):
    os.system("git --git-dir=./ashrae/.git pull")
else:
    os.system("git clone https://github.com/strongio/ashrae")
# os.system("pip uninstall ./ashrae")
!pip install ./ashrae

!pip install git+https://github.com/strongio/torch-kalman@dev#egg=torch_kalman

# set directories:
os.environ['PROJECT_ROOT'] = "/content/drive/My Drive/ashrae"
os.environ['DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT'], "data")

# copy repo notebooks to current dir:
for dirname, _, filenames in os.walk('/content/ashrae/notebooks'):
    for filename in filenames:
        copyfile(src= os.path.join(dirname, filename), dst= filename)

"""
function ClickConnect(){
console.log("Working"); 
document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect,60000)
"""