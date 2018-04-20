1.  Install [Git](https://git-scm.com/downloads)
2.  Install the Python data science platform
    [Anaconda](https://www.anaconda.com/download/) (choose the **Python
    3.6** version)

On Windows:

To be able to run all shell commands/edit 

1.  Install the [GNU CoreUtils for
    Windows](http://gnuwin32.sourceforge.net/packages/coreutils.htm)
    (after installation ensure that the installation path is added to
    the Path environment variable)
2.  Install the [GNU grep for
    Windows](http://gnuwin32.sourceforge.net/packages/grep.htm) (after
    installation ensure that the installation path is added to the Path
    environment variable)
3.  Install [Vim](http://www.vim.org/) (or another editor of your choice
    that can display diffs side-by-side)
4.  Optional: when starting an Anaconda prompt change the output code
    page to UTF-8 by entering the command \"chcp 65001\"

### Python 3.5 environment 

#### Create a new Python 3.5 environment

1.  Launch \"Anaconda Prompt\" (this assumes you have an Anaconda Python
    3.6 version installed - see above)
2.  Enter the command \"conda create -n py35 python=3.5 anaconda\"
3.  Close the Anaconda Prompt
4.  Launch \"Anaconda Prompt (py35)\"

#### Install Keras and supporting libraries

1.  Enter \"conda install keras\" (if the conda command is not available
    you might have to add \<your home directory\>\\Anaconda3\\Scripts to
    your path)

#### To plot graphs in Keras

1.  Install [Graphviz](http://www.graphviz.org/) for your platform
    -   On Windows add the path to the Graphviz binaries
        (e.g. C:\\Program Files (x86)\\Graphviz2.38\\bin)  to your PATH
        environment variable (see
        [also](https://stackoverflow.com/questions/36869258/how-to-use-graphviz-with-anaconda-spyder))
2.  \"conda install graphviz\"
3.  \"conda install pydot\"
4.  \"conda install pydotplus\"

#### To save trained models in Keras 

1.  \"conda install hdf5\"
2.  \"conda install h5py\"

#### Emitting Unicode (UTF-8) in the Anaconda prompt

1.  When starting an Anaconda prompt change the output code page to
    UTF-8 by entering the command \"chcp 65001\"
2.  Enter the command \"set PYTHONIOENCODING=UTF-8\" to ensure that
    Python emits UTF-8 (this can also be automated for the [Anaconda
    prompt](https://conda.io/docs/user-guide/tasks/manage-environments.html#windows))
