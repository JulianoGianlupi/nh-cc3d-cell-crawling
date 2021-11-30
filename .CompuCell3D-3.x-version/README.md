# Script for deploying CompuCell3D simulations as tools in nanoHUB

This script was made to make the generation of GitHub repositories to deploy CompuCell3D tools an easier task.

## Note:

The current version of CompuCell3D in nanoHUB is 3.7.6

## Requirements:
* Python 3
* Be able to run ```python``` from the command line. You can 1) edit your PATH system variable, 2) provide the full path command, or 3) create an alias.
* GitHub account
* NanoHub account

###### Posible requirement:

* Run ```git``` from the command line


## How to use:

1. Dowload / Clone this repository to your machine.
1. Create a GitHub repository where you're going to have your nanoHub tool ("tool repo"). Have it cloned to your computer.
1. On a command line from the cc3d-nanoHub-settuper directory on your machine call the script ```tool_maker.py```. If successfull it'll copy all the necessary files for the nanoHub tool. You will provide three arguments to the script:
```
python tool_maker.py <short tool name> <full-path-to-new-tool-directory> <full-path-to-your-cc3d-project>
```
The ```<short tool name>``` needs to be between 3 and 15 alphanumeric characters without spaces. The ```<full-path-to-your-cc3d-project>``` is where the ```.cc3d``` of your CompuCell3D simulation is. ```<full-path-to-new-tool-directory>``` is the tool repo.

Some files might have lost run permissions. You need to check two files in your tool repo:
1. ```invoke``` file in the ```middleware``` subdirectory,
1. And the ```.sh``` file in the ```bin``` subdirectory.

To check if they still have the executable permission you should go into the tool repo in the GitHub website:

[![.sh executable](https://i.imgur.com/9zNpYF0.png ".sh executable")](https://i.imgur.com/9zNpYF0.png ".sh executable")

[![invoke executable](https://i.imgur.com/IH6aEIl.png "invoke executable")](https://i.imgur.com/IH6aEIl.png "invoke executable")

If any of them is not executable you should, from a command line, go to the relevant directory and issue the following commands:

```
$ git update-index --chmod=+x invoke
$ git commit -m "Changing file permissions"
$ git push
```
Changing the first command accordinly if you're dealing with the ```.sh``` file.


## nanoHub deployement:

Go to ```https://nanohub.org/tools/create``` and follow the instructions there. The one thing you **must** be carefull about is that the ```<short tool name>``` you gave before is **the same** as the one you input on the first box (```Tool Name```) on nanoHub's tool creation page.
