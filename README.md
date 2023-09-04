# AIMLv2-demos

A few helpful things for the demos in this series:

Demo1: demonstrates how supervised learning is used to train a neural network based on training data that roughly matches a mathematical function f(x) = sin(x) + 0.1x^2

Demo 2: demonstrates how hyperparameters can change the performance of machine learning. In Demo2a the learning rate hyperparameter is reduced to show how gradient descent convergence can an excessively long time. In Demo2b, the learning rate is increased, making the gradient descent too large. Although training progresses much faster, it eventually destablizes and does not train the model correctly.

Demo3: Shows how SVM can be used for complex classification problems. In 3a we show a 2D rendering of 2 classes that appear to be concentric circles. In 3b we use rbf to project into 3D so a hyperplane can be used to identify the kernel and seperate the classes

Demo4: is contained in the directory "Demo4-TicTacToe". This is a demonstration of Reinforcement Learning using Q-learning. It is based on the following repo: https://github.com/Rohithkvsp/Tic-Tac-Toe-Reinforcement-learning.git​. First, run the script: "Play Dumb Agent.py". Then train with Q-learning using "Train.py". Then rerun the game with "Play Q-Learning.py" to see how the agent has been trained to play.

Demo 5: Shows how a simple corpus can be trained using Word2vec. The corpus is contained in "Demo5-Word2Vec.py". This script saves the trained model as "Demo5-word2vec_model.bin". Run the script "Demo5-Word-Embedding-3D.py" which loads the trained model, then uses PCA to reduce to 3 dimensions, then renders it as a 3D plot.

Demo 6: Two possible demos how to use Tensorboard. Demo6a is a simple demo of supervised learning to learn training data that roughly models a polynomial function. Demo6b is a 3-layer CNN. In either case, the script saves the data to the /logs folder, which is used by Tensorboard. To launch Tensorboard, use this command: #python3 -m tensorboard.main --logdir=logs/train, where "logs/train" is the folder where data is stored. Then simply open a browser and go to http://localhost:6006 to view tensorboard. Note, if there are path problems with Tensorboard, you can set an alias using this command: alias tensorboard='python3 -m tensorboard.main'

Demo 7: 
#Add Jupyter to the path: 
#export PATH=$HOME/Library/Python/3.9/bin:$PATH%  
