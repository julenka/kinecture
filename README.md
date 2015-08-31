Kinecture aims to analyze conversation dynamics in the classroom. This project aims to help determine what is happening at each timestep in a classroom session.

### Check Out What I Did!
So far I've just done some data exploration, which you can view in the browser, [here](https://github.com/julenka/kinecture/blob/master/data_exploration.ipynb)!

### Install Instructions
You don't have to install Python to view the experiments, just check out the .ipynb files in this repo from your browser to get an idea.

But eventuall you'll probably want to modify things, which means you'll need to install stuff. I am using an IPython 
notebook for this since it's a great way to share experiments! Also I'm using a pretty
standard set of tools for doing data science in Python. Here's how to get set up using Homebrew. Assumes OSX:

```
# install homebrew, see http://brew.sh/
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# install python3, ipython
brew install python3
brew install pip3

# install packages
pip3 install ipython
pip3 install scipy
pip3 install pandas
pip3 install xlrd
pip3 install seaborn
pip3 install sklearn
```

