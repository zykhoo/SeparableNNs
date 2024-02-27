# SeparableNNs
Separable Hamiltonian Neural Networks

The modelling of dynamical systems from discrete observations is a challenge faced by modern scientific and engineering data systems. Hamiltonian systems are one such fundamental and ubiquitous class of dynamical systems. Hamiltonian neural networks are state-of-the-art models that unsupervised-ly regress the Hamiltonian of a dynamical system from discrete observations of its vector field under the learning bias of Hamilton's equations. 
Yet Hamiltonian dynamics are often complicated, especially in higher dimensions where the state space of the Hamiltonian system is large relative to the number of samples. A recently discovered remedy to alleviate the complexity between state variables in the state space is to leverage the additive separability of the Hamiltonian system and embed that additive separability into the Hamiltonian neural network. 
Following the nomenclature of physics-informed machine learning, we propose three \textit{separable Hamiltonian neural networks}. These models embed additive separability within Hamiltonian neural networks. The first model uses additive separability to quadratically scale the amount of data for training Hamiltonian neural networks. The second model embeds additive separability within the loss function of the Hamiltonian neural network. The third model embeds additive separability through the architecture of the Hamiltonian neural network using conjoined multilayer perceptions. We empirically compare the three models against state-of-the-art Hamiltonian neural networks, and demonstrate that the separable Hamiltonian neural networks, which alleviate complexity between the state variables, are more effective at regressing the Hamiltonian and its vector field.




