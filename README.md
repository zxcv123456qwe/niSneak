



# **About niSNEAK**
This repository is the official implementation of  niSNEAK

 In software analytics,   better results can be found using less information by looking   closely at the data before leaping to new conclusions. For example, niSNEAK  is a  new hyperparameter optimizer   that   configures learners  to   predict for future values of project health indicators such as 
{C,I,R} where: 
 * C= number of commits; 
 * I=number of closed issues, and  
 * R=number of closed pull requests.  

Instead of blindly applying a sequence of learners, niSNEAK reflects on the landscape of the data. Specifically niSNEAK (a) recursively partitions the landscape of possible configurations   (via random projections), then (b) uses an weighted  analysis to make decisions that most divides the landscape; then (c)     samples a small number of points  cross the surviving parts of the landscape. 



When compared to the prior state-of-the-art in hyperparameter optimization, niSNEAK's predictions for $\{C,I,R\}$, one year in the future,  are far more accurate than  the predictions made by the prior state of the art (FLASH, HYPEROPT, OPTUNA, and differential evolution). For example,  median    errors seen in 20 runs with OPTUNA and niSNEAK were 
 * OPTUNA=\{C=149\%, I=61\%, R=119\%\}\ and 
* niSNEAK=\{C=47\%, I=33\%, R=0\%\}    

(here, lower values are better).

In support of our "look before your leap" hypothesis, we note that niSNEAK can find better configurations after looking  at around 100 configurations while other tools needed to explore 3500, and 5000 (for OPTUNA, HYPEROPT). We suggest that, for future work, if could be insightful and useful to apply ``look before you leap'' to other problems in software analytics. 




## **Reproducing Results** 

### To obtain the results for niSNEAK as well as the baselines:

### niSNEAK:

 * install dependencies for niSNEAK availabe through our requirements.txt folder in the ``./niSNEAK/`` directory.
 * ``pip install -r requirements.txt``

  * run the python file named ``rq1.py`` under ``./niSNEAK/rq1.py``
  * ``python rq1.py``



### Baselines:

For running the majority of our baselines open the Baselines directory and then in the ``./baselines/nue_framework/src`` directory run the following commands:
 * ``pip install -r requirements.txt``
 * ``python ml.py see``

#### Original [NUE repository](https://github.com/lyonva/Nue)

To repeat the experiment this must be executed 20 times.

Make sure to install the requirements for nue available through their requirements.txt folder under the same directory.

To run OPTUNA and HYPEROPT please run all of the available jupyter notebooks on each directory as is:
Prior to that make sure to run:

 * ``pip install optuna``
 * ``pip install hyperopt``

#### OPTUNA:

 * ``./baselines/optuna/Optuna commits.ipynb``
 * ``./baselines/optuna/Optuna issues.ipynb``
 * ``./baselines/optuna/Optuna prs.ipynb``

#### HYPEROPT:

 * ``./baselines/hyperopt/Hyperopt commits.ipynb``
 * ``./baselines/hyperopt/Hyperopt issues.ipynb``
 * ``./baselines/hyperopt/Hyperopt prs.ipynb``

