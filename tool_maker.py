# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:20:26 2019

@author: jferrari
"""
import os
import shutil
import sys
import stat

num_args = len(sys.argv)
if num_args != 4:
    print("Usage: python %s <short-tool-name> <full-path-to-new-directory> <full-path-to-your-cc3d-project>" % (
        os.path.basename(__file__)))
    print("Short tool name should be unique and contain 3-15 \nalphanumeric characters, no spaces.")
    print("Once you register your tool, you cannot change the short tool name, \nso be careful to pick a good one.")
    print("If you have spaces in your path put it inside.")

    sys.exit(1)

shortName = sys.argv[1]

if ((len(shortName) > 15) or
        (len(shortName) < 3) or
        (' ' in shortName) or
        (not shortName.isalnum())):
    print("Invalid <short-tool-name> or wrong number or arguments:\n  ")
    print("Short tool name should be unique and contain 3-15 \nalphanumeric characters, no spaces.")
    print("Once you register your tool, you cannot change the short tool name, \nso be careful to pick a good one.")
    print("If you have spaces in your path put it inside.")

    sys.exit(1)

dest = sys.argv[2]
source = sys.argv[3]

if not os.path.isdir(source):
    raise Exception('Path of source not found:', source)
else:
    if 'Simulation' not in os.listdir(source):
        raise Exception('Simulation folder not found in:', source)
    else:
        found_cc3d = False
        for name in os.listdir(source):
            if name.endswith('.cc3d'):
                found_cc3d = True
                cc3d_file_name = name.replace('.cc3d', '')
        if not found_cc3d:
            raise Exception('.cc3d file not found in:', source)

if not os.path.isdir(dest):
    # raise Exception('Path of destination not found:', dest)
    print('Path of destination not found:', dest)
    no_dest = True
    if no_dest:
        while no_dest:
            print('Do you wish for\n', dest)
            create = input('\nto be created by this program?[y/n]')
            if create == 'n' or create == 'N' or create == 'y' or create == 'Y':
                no_dest = False
        if create == 'n' or create == 'N':
            raise Exception('Path of destination does not exist:', dest)
        else:
            os.makedirs(dest)

print("\n\n STEP 1: copying your simulation files\n")
tool_cc3d_files = os.path.join(dest, 'main')
print(source, '-->', tool_cc3d_files)
try:
    shutil.copytree(source, tool_cc3d_files,
                    ignore=shutil.ignore_patterns('*.pyc', '*.zip', '*.vtk', '*.png', '*.jpeg', '*.jpg', '__pycache__'))
except:
    print("unable to copy")

print("\n\n STEP 2: copying critical tool files\n")
for rel_name in os.listdir(os.path.dirname(__file__)):
    name = os.path.join(os.path.dirname(__file__), rel_name)
    print(name)
    try:
        if rel_name[0] != '.' and rel_name != '__pycache__':
            if os.path.isdir(name):
                source_dir = name
                dest_dir = os.path.join(dest, rel_name)
                print(source_dir, '-->', dest_dir)
                shutil.copytree(source_dir, dest_dir)
            elif not (rel_name == sys.argv[0] or rel_name == "README.md" or rel_name == "tool_maker.py"):
                source_file = name
                dest_file = os.path.join(dest, rel_name)
                print(source_file, '-->', dest_file)
                shutil.copy(source_file, dest_file)
    except:
        print("unable to copy")

print("\n\n STEP 3: renaming files and content\n")

os.chdir(dest)

with open('middleware/invoke', 'r') as f:
    new_text = f.read().replace("-t toolname", "-t " + shortName)
    new_text = new_text.replace("@tool/bin/toolname.sh", "@tool/bin/" + shortName + ".sh")

with open('middleware/invoke', 'w') as f:
    f.write(new_text)

f_st = os.stat('middleware/invoke')
os.chmod('middleware/invoke', f_st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

f_st = os.stat('bin/cc3d_count.sh')
os.chmod('bin/cc3d_count.sh', f_st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


old_sh = os.path.join('bin', 'nh-cc3d-toolname.sh')
new_sh = os.path.join('bin', shortName + '.sh')

try:
    shutil.move(old_sh, new_sh)
    print('renamed', old_sh, new_sh)
    os.chmod(new_sh, 0o777)
    f_st = os.stat(new_sh)
    os.chmod(new_sh, f_st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
except:
    print("couldn't rename .sh file in bin, please do it manually")

with open(new_sh, 'r') as f:
    old_text = f.read()
    new_text = old_text.replace("cc3dFileName", cc3d_file_name)
    new_text = new_text.replace("toolName", shortName)

with open(new_sh, 'w') as f:
    f.write(new_text)
    print("updated ", new_sh)
