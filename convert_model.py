import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.tree import _tree

class RandomForestNet(nn.Module):
    def __init__(self, n_estimators, max_depth, n_features):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_features = n_features
        # Camadas para cada árvore
        self.trees = nn.ModuleList([self._create_tree_layers(max_depth) 
                                  for _ in range(n_estimators)])
        
    def _create_tree_layers(self, max_depth):
        layers = []
        # Camada de entrada
        layers.append(nn.Linear(self.n_features, 1))
        layers.append(nn.ReLU())
        
        # Camadas intermediárias
        for i in range(max_depth - 1):
            layers.append(nn.Linear(1, 1))
            layers.append(nn.ReLU())
            
        # Camada de saída
        layers.append(nn.Linear(1, 1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        outputs = [tree(x) for tree in self.trees]
        return torch.mean(torch.stack(outputs), dim=0)

def tree_to_weights(tree):
    """Converte uma árvore de decisão para pesos de rede neural"""
    tree_ = tree.tree_
    weights = []
    
    # Extrai as características da árvore
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    feature = tree_.feature
    threshold = tree_.threshold
    
    # Mapeia nós para camadas
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    
    stack = [(0, 0)]  # (node_id, depth)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        
        # Se é um nó de decisão
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
            
    # Calcula o número máximo de nós por nível
    max_nodes_per_level = [0] * (node_depth.max() + 1)
    for depth in node_depth:
        max_nodes_per_level[depth] += 1
        
    # Cria pesos baseados nas regras
    for i in range(n_nodes):
        if not is_leaves[i]:
            # Peso para feature
            w = np.zeros(tree_.n_features)
            w[feature[i]] = 1.0
            weights.append(w)
            
            # Bias (threshold)
            weights.append(np.array([-threshold[i]]))
            
            # Adiciona pesos para nós filhos
            if children_left[i] != -1:
                # Peso para feature do filho esquerdo
                w_left = np.zeros(tree_.n_features)
                w_left[feature[children_left[i]]] = 1.0
                weights.append(w_left)
                
                # Bias (threshold) do filho esquerdo
                weights.append(np.array([-threshold[children_left[i]]]))
                
                # Peso para feature do filho direito
                w_right = np.zeros(tree_.n_features)
                w_right[feature[children_right[i]]] = 1.0
                weights.append(w_right)
                
                # Bias (threshold) do filho direito
                weights.append(np.array([-threshold[children_right[i]]]))
                
    # Adiciona pesos finais para camada de saída
    weights.append(np.ones(1))  # Peso final
    weights.append(np.zeros(1))  # Bias final
    
    return weights

# Carrega o modelo antigo
old_model = joblib.load('model.joblib')

# Verifica características do modelo
n_features = old_model.n_features_in_
if n_features is None:
    raise ValueError("Não foi possível determinar o número de features do modelo")

# Cria novo modelo PyTorch
new_model = RandomForestNet(
    n_estimators=old_model.n_estimators,
    max_depth=old_model.max_depth,
    n_features=n_features
)

# Converte cada árvore
for i, tree in enumerate(old_model.estimators_):
    weights = tree_to_weights(tree)
    
    # Verifica compatibilidade de pesos
    if len(weights) != len(list(new_model.trees[i].parameters())):
        raise ValueError(f"Incompatibilidade de pesos na árvore {i}")
        
    # Aplica os pesos convertidos
    with torch.no_grad():
        for param, weight in zip(new_model.trees[i].parameters(), weights):
            if param.shape != weight.shape:
                # Redimensiona o peso se necessário
                weight = weight.reshape(param.shape)
            param.copy_(torch.from_numpy(weight).float())

# Salva o novo modelo
torch.save(new_model.state_dict(), 'model.pth')
print("Modelo convertido com sucesso para model.pth")
