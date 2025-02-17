import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron as SklearnPerceptron
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Perceptron Iris", layout="wide")

# =====================================
# BARRE LATÉRALE
# =====================================
with st.sidebar:
    st.header("⚙️ Paramètres du Modèle")
    
    # Contrôles des hyperparamètres
    learning_rate = st.slider("Taux d'apprentissage", 0.001, 1.0, 0.15, 0.01)
    n_iters = st.slider("Nombre d'itérations", 10, 200, 50)
    test_size = st.slider("Taille du set de test", 0.1, 0.5, 0.25, 0.05)
    
    # Sélection des caractéristiques
    st.subheader("Sélection des Features")
    feature_options = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    feature1 = st.selectbox("Feature 1", feature_options, index=0)
    feature2 = st.selectbox("Feature 2", feature_options, index=3)
    
    # Test manuel
    st.subheader("🔍 Test Manuel")
    sepal_length = st.number_input("Longueur du sépale (cm)", 4.0, 7.0, 5.1)
    petal_width = st.number_input("Largeur du pétale (cm)", 0.1, 2.5, 0.3)

# =====================================
# CONTENU PRINCIPAL
# =====================================
st.title("🌷 Classification des Iris avec Perceptron")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('Iris.csv')
    df = df[df['Species'].isin(['Iris-setosa', 'Iris-versicolor'])]
    return df

df = load_data()
@st.cache_data
def load_data2():
    df = pd.read_csv('Iris.csv')
    df = df[df['Species'].isin(['Iris-setosa', 'Iris-versicolor','Iris-virginica'])]
    return df
df2 = load_data2()

# Préparation des données
X = df[[feature1, feature2]].values
y = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1}).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, stratify=y)

# Implémentation du Perceptron
class MyPerceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.errors_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.n_iters):
            total_error = 0
            for xi, yi in zip(X, y):
                prediction = self.activation(xi)
                error = yi - prediction
                update = self.lr * error
                self.weights += update * xi
                self.bias += update
                total_error += int(error != 0)
            
            self.errors_history.append(total_error)
            if total_error == 0:
                break
                
    def activation(self, x):
        return 1 if (np.dot(x, self.weights) + self.bias) >= 0 else 0
    
    def predict(self, X):
        return np.array([self.activation(x) for x in X])

# Entraînement
perceptron = MyPerceptron(learning_rate=learning_rate, n_iters=n_iters)
perceptron.fit(X_train, y_train)

# =====================================
# VISUALISATIONS
# =====================================
col1, col2 = st.columns(2)

with col1:
    # Visualisation 3D interactive
    st.subheader("📊 Exploration 3D des Données")
    fig_3d = px.scatter_3d(df2, 
                          x=feature1, 
                          y=feature2,
                          z='PetalLengthCm',
                          color='Species',
                          hover_name='Species',
                          title="Distribution des Espèces en 3D")
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    # Visualisation de la convergence
    st.subheader("📈 Courbe de Convergence")
    fig, ax = plt.subplots()
    ax.plot(perceptron.errors_history, marker='o', color='purple', label='Erreurs')
    ax.set_xlabel("Époques")
    ax.set_ylabel("Erreurs de Classification")
    ax.set_title("Évolution des Erreurs pendant l'Entraînement")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Commentaire interactif
    if perceptron.errors_history[-1] == 0:
        st.success("✅ Le modèle a convergé avec succès !")
    elif len(perceptron.errors_history) == n_iters:
        st.warning("⚠️ Le modèle n'a pas convergé. Essayez d'augmenter le nombre d'itérations ou de réduire le taux d'apprentissage.")
    else:
        st.info("🔍 Le modèle est en train d'apprendre. Observez la courbe pour ajuster les paramètres.")

# Frontière de décision 3D
st.subheader("🎨 Frontière de Décision 3D")
xx, yy = np.meshgrid(np.linspace(X_scaled[:,0].min()-1, X_scaled[:,0].max()+1, 50),
                     np.linspace(X_scaled[:,1].min()-1, X_scaled[:,1].max()+1, 50))

Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig = go.Figure(data=[
    go.Surface(x=xx, y=yy, z=np.zeros_like(xx)),
    go.Scatter3d(
        x=X_scaled[:,0],
        y=X_scaled[:,1],
        z=y,
        mode='markers',
        marker=dict(
            size=5,
            color=y,
            colorscale='Viridis'
        )
    )
])

fig.update_layout(scene=dict(
    xaxis_title=feature1,
    yaxis_title=feature2,
    zaxis_title='Classe',
    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
))
st.plotly_chart(fig, use_container_width=True)

# =====================================
# ÉVALUATION
# =====================================
tab1, tab2, tab3 = st.tabs(["📋 Performance", "🎯 Matrice de Confusion", "🔬 Comparaison"])

with tab1:
    y_pred = perceptron.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    
    # Affichage des paramètres
    st.write("**Configuration du modèle:**")
    params = {
        "Features utilisées": f"{feature1} & {feature2}",
        "Taux d'apprentissage": learning_rate,
        "Itérations effectuées": len(perceptron.errors_history),
        "Taille du set d'entraînement": f"{len(X_train)} échantillons"
    }
    st.json(params)

with tab2:
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm,
                      labels=dict(x="Prédit", y="Réel", color="Count"),
                      x=['Setosa', 'Versicolor'],
                      y=['Setosa', 'Versicolor'],
                      text_auto=True)
    st.plotly_chart(fig_cm, use_container_width=True)

with tab3:
    # Comparaison avec scikit-learn
    sk_perceptron = SklearnPerceptron(alpha=learning_rate, max_iter=n_iters)
    sk_perceptron.fit(X_train, y_train)
    sk_acc = accuracy_score(y_test, sk_perceptron.predict(X_test))
    
    col1, col2 = st.columns(2)
    col1.metric("Notre Perceptron", f"{accuracy_score(y_test, y_pred):.2%}")
    col2.metric("Scikit-learn", f"{sk_acc:.2%}")

# =====================================
# PRÉDICTION MANUELLE
# =====================================
st.subheader("🔮 Prédiction en Temps Réel")
input_data = np.array([[sepal_length, petal_width]])
input_scaled = scaler.transform(input_data)
prediction = perceptron.predict(input_scaled)[0]

col1, col2 = st.columns(2)
col1.write(f"### Valeurs entrées :")
col1.write(f"- {feature1}: {sepal_length} cm")
col1.write(f"- {feature2}: {petal_width} cm")

col2.write(f"### Résultat :")
col2.markdown(f"<h2 style='color: {'green' if prediction == 0 else 'red'};'>"
             f"{'Setosa' if prediction == 0 else 'Versicolor'}</h2>", 
             unsafe_allow_html=True)