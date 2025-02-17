import numpy as np  

def activation(x):
    # Fonction d'activation de type seuil : renvoie 1 si x >= 0, sinon 0.
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_features = X.shape[1]
        # Initialisation des poids à zéro
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Boucle d'apprentissage
        for _ in range(self.epochs):
            for i in range(len(X)):
                # Calcul de la somme pondérée + biais
                linear_output = np.dot(X[i], self.weights) + self.bias
                # Application de la fonction d'activation
                y_pred = activation(linear_output)
                # Calcul de l'erreur
                error = y[i] - y_pred
                # Mise à jour des poids et du biais
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

    def predict(self, X):
        # Prédiction pour chaque entrée
        return [activation(np.dot(x, self.weights) + self.bias) for x in X]

# Exemple d'entraînement pour la fonction AND
X = np.array([[1,1],[0,0], [0,1], [1,0], [1,1]])
y = np.array([1, 0, 0, 0, 1])  # Fonction AND : seule la combinaison (1,1) donne 1

p = Perceptron()
p.train(X, y)
print(p.predict(X))
print(p.predict([[1,1]]))

