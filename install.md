# Installing the required software

These instructions should get you from zero to a working setup on Windows,
Mac and Linux.

In general terms you will need a terminal, git, python3, and jupyter. If you
know what all these mean you probably do not need to follow these instructions.

👋 If you get stuck with these instructions please create an issue
[here](https://github.com/mdeff/ntds_2017/issues/new).

The instructions were taken directly from the amazing people at
[Software Carpentry](https://software-carpentry.org).

# Windows

### git & bash

* Download [the git for Windows installer](https://git-for-windows.github.io/).
* Run the installer and follow the steps bellow:
  1. Click on `Next`.
  1. Click on `Next`.
  1. Keep `Use Git from the Windows Command Prompt` selected and click on `Next`.
     Some software won't work if you forget to do this.
	 Please run the installer again if it happens.
  1. Click on `Next`.
  1. Keep `Checkout Windows-style, commit Unix-style line endings` selected and click on `Next`.
  1. Keep `Use Windows' default console window` selected and click on `Next`.
  1. Click on `Install`.
  1. Click on `Finish`.
* If your `HOME` environment variable is not set (or you don't know what this is):
  1. Open the command prompt (open the Start Menu then type cmd and press `Enter`).
  1. Type the following line into the prompt window: `setx HOME "%USERPROFILE%"`.
  1. Press `Enter`, you should see `SUCCESS: Specified value was saved`.
  1. Quit the command prompt by typing `exit` then pressing `Enter`.

This will provide you with both git and bash in the Git Bash program.

### python

1. Open <https://www.anaconda.com/download> with your web browser.
1. Download the Python 3 installer for Windows.
1. Install Python 3 using all the defaults except make sure
   to check `Make Anaconda the default Python`.

# Mac

### bash

Bash is the default shell in all versions of Mac OS X, no need to install
anything. You access bash from the Terminal (found in
`/Applications/Utilities`). See the Git installation [video
tutorial](https://www.youtube.com/watch?v=9LQhwETCdwY) for an example on how to
open the Terminal. You may want to keep Terminal in your dock for fast access.

### git

* For OS X 10.9 and higher, install Git for Mac by downloading and running the most recent "mavericks" installer from [here][git_mac].
* For older versions of OS X (10.5-10.8), use the most recent "snow-leopard" installer from [here][git_mac].

Note that there will not be anything in your /Applications folder as git is a command line program.

[git_mac]: http://sourceforge.net/projects/git-osx-installer/files/

### python

1. Open <https://www.anaconda.com/download> with your web browser.
1. Download the Python 3 installer for OS X.
1. Install Python 3 using all the defaults.

# Linux

### bash

The default shell is usually bash, but if your machine is set up differently
you can run it by opening a terminal and typing bash.

### git

If Git is not already available on your machine you can try to install it via
your distro's package manager, e.g. `sudo apt-get install git` for
Debian/Ubuntu/Mint and `sudo yum install git` for Fedora.

### python

1. Open <https://www.anaconda.com/download> with your web browser.
1. Download the Python 3 installer for Linux.
1. Install Python 3 using all the defaults.
