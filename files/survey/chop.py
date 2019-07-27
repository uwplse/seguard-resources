from msbase.subprocess_ import try_call_std

import glob
import os
import shutil

for png in glob.glob("*/*.png"):
    if "seguard" in png:
        continue
    try_call_std(["convert", png, "-crop", "400x200", "new.png"])
    shutil.copyfile("new-0.png", png.replace(".png", ".vt-small.png"))
    os.system("rm new-*.png")