## SGL: Scalable Graph Learning

**SGL** is a Graph Neural Network (GNN) toolkit targeting scalable graph learning, which supports deep graph learning on extremely large datasets. SGL allows users to easily implement scalable graph neural networks and evalaute its performance on various downstream tasks like node classification, node clustering, and link prediction. Further, SGL supports auto neural architecture search functionality based on <a href="https://github.com/PKU-DAIR/open-box" target="_blank" rel="nofollow">OpenBox</a>. SGL is designed and developed by the graph learning team from the <a href="https://cuibinpku.github.io/index.html" target="_blank" rel="nofollow">DAIR Lab</a> at Peking University.



## Library Highlights

+ **High scalability**: Follow the scalable design paradigm **SGAP** in <a href="https://arxiv.org/abs/2203.00638" target="_blank" rel="nofollow">PaSca</a>, SGL scale to graph data with billions of nodes and edegs.
+ **Auto neural architecture search**: Automatically choose decent neural architectures according to specific tasks, and pre-defined objectives (e.g., inference time).
+ **Ease of use**: User-friendly interfaces of implementing existing scalable GNNs and executing various downstream tasks.



## Installation (TODO)

#### Install from pip


#### Install from source




## Quick Start (TODO)
**TODO**
A quick start example is given by:
```python
import torch

from SGL.dataset import Planetoid
from SGL.models.homo import SGC
from SGL.tasks import NodeClassification

dataset = Planetoid("pubmed", "./", "official")
model = SGC(prop_steps=3, feat_dim=dataset.num_features, num_classes=dataset.num_classes)

device = "cuda:0"
test_acc = NodeClassification(dataset, model, lr=0.1, weight_decay=5e-4, epochs=200, device=device).test_acc
print(test_acc)
```

**TODO**
An example of the auto neural network search functionality is as follows:
```python
import torch

from SGL.dataset import Planetoid
from SGL.models.search_models import SearchModel
from SGL.search.auto_search import SearchManager

from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter

dataset = Planetoid("cora", "./", "official")

def AutoSearch(arch):
    model = SearchModel(arch, dataset.num_features, int(dataset.num_classes), 64)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    acc_res, time_res = SearchManager(dataset, model, lr=0.01, weight_decay=5e-4, epochs=200, device=device)._execute()
    result = dict()
    result['objs'] = np.stack([-acc_res, time_res], axis=-1)
    return result

# Test arbitrary configuration
res = AutoSearch([2, 0, 1, 2, 3, 0, 0])
print(res)

# Define configuration space
config_space = ConfigurationSpace()
prop_steps = UniformIntegerHyperparameter("prop_steps", 1, 10, default_value=3)
prop_types = UniformIntegerHyperparameter("prop_types", 0, 1)
mesg_types = UniformIntegerHyperparameter("mesg_types", 0, 8, default_value=2)
num_layers = UniformIntegerHyperparameter("num_layers", 1, 10, default_value=2)
post_steps = UniformIntegerHyperparameter("post_steps", 1, 10, default_value=0)
post_types = UniformIntegerHyperparameter("post_types", 0, 1)
pmsg_types = UniformIntegerHyperparameter("pmsg_types", 0, 5)
config_space.add_hyperparameters([prop_steps, prop_types, mesg_types, num_layers, post_steps, post_types, pmsg_types])

def SearchTarget(config_space):
    arch = [2, 0, 1, 2, 3, 0, 0]
    arch[0] = config_space['prop_steps']
    arch[1] = config_space['prop_types']
    arch[2] = config_space['mesg_types']
    arch[3] = config_space['num_layers']
    arch[4] = config_space['post_steps']
    arch[5] = config_space['post_types']
    arch[6] = config_space['pmsg_types']
    result = AutoSearch(arch)
    return result

dim = 7
bo = SMBO(SearchTarget,
          config_space,
          num_objs=2,
          num_constraints=0,
          max_runs=3500,
          surrogate_type='prf',
          acq_type='ehvi',
          acq_optimizer_type='local_random',
          initial_runs=2*(dim+1),
          init_strategy='sobol',
          ref_point=[-1, 0.00001],
          time_limit_per_trial=5000,
          task_id='quick_start',
          random_state=1)
history = bo.run()
print(history)
```


## Related Publications

**PaSca: a Graph Neural Architecture Search System under the Scalable Paradigm** Wentao Zhang, Yu Shen, Zheyu Lin, Yang Li, Xiaosen Li, Wen Ouyang, Yangyu Tao, Zhi Yang, and Bin Cui; The world wide web conference (WWW 2022, CCF-A). https://arxiv.org/abs/2203.00638



## License

The entire codebase is under [MIT license](LICENSE).
