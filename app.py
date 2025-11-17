import gradio as gr
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Charger le modèle et les fichiers
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def create_feature_importance_plot():
    """Visualisation de l'importance des features"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], color='steelblue')
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Importance des Caractéristiques dans la Prédiction')
    plt.tight_layout()
    
    output_path = 'plots/feature_importance.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path

def create_input_summary_plot(inputs):
    """Résumé visuel des inputs utilisateur"""
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(feature_names))
    plt.barh(feature_names, inputs, color=colors)
    plt.xlabel('Valeur')
    plt.title('Résumé des Caractéristiques de la Maison')
    plt.tight_layout()
    
    output_path = 'plots/input_summary.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path

def create_prediction_plot(prediction, lower, upper):
    """Visualisation de la prédiction avec intervalle de confiance"""
    plt.figure(figsize=(10, 6))
    
    plt.barh(['Prix Prédit'], [prediction], color='green', alpha=0.7, label='Prédiction')
    plt.errorbar([prediction], ['Prix Prédit'], 
                 xerr=[[prediction - lower], [upper - prediction]],
                 fmt='none', color='red', capsize=10, capthick=2,
                 label='Intervalle de confiance 95%')
    
    plt.xlabel('Prix ($)')
    plt.title('Prédiction du Prix avec Intervalle de Confiance')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = 'plots/prediction.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path

def predict_price(square_feet, bedrooms, bathrooms, age_years, 
                 lot_size, garage_spaces, neighborhood_score):
    """Fonction principale de prédiction"""
    
    # Préparer les inputs
    inputs = np.array([[square_feet, bedrooms, bathrooms, age_years,
                       lot_size, garage_spaces, neighborhood_score]])
    
    # Normaliser
    inputs_scaled = scaler.transform(inputs)
    
    # Prédiction
    prediction = model.predict(inputs_scaled)[0]
    
    # Calcul de l'intervalle de confiance (95%)
    predictions_trees = np.array([tree.predict(inputs_scaled)[0] 
                                 for tree in model.estimators_])
    std = np.std(predictions_trees)
    lower_bound = prediction - 1.96 * std
    upper_bound = prediction + 1.96 * std
    
    # Créer les visualisations
    feature_plot = create_feature_importance_plot()
    input_plot = create_input_summary_plot(inputs[0])
    prediction_plot = create_prediction_plot(prediction, lower_bound, upper_bound)
    
    # Texte de résultat
    result_text = f"""
    ### 🏠 Prédiction du Prix
    
    **Prix estimé : ${prediction:,.2f}**
    
    **Intervalle de confiance à 95% :**
    - Minimum : ${lower_bound:,.2f}
    - Maximum : ${upper_bound:,.2f}
    
    La prédiction est basée sur un modèle Random Forest entraîné sur 1000 maisons.
    """
    
    return result_text, feature_plot, input_plot, prediction_plot

# Interface Gradio
with gr.Blocks(title="Prédicteur de Prix Immobiliers") as demo:
    gr.Markdown("#  Prédicteur de Prix Immobiliers")
    gr.Markdown("Estimez le prix d'une maison en fonction de ses caractéristiques")
    
    with gr.Row():
        with gr.Column():
            square_feet = gr.Slider(800, 4000, value=2000, step=100, 
                                   label="Surface (pieds carrés)")
            bedrooms = gr.Slider(1, 5, value=3, step=1, 
                               label="Nombre de chambres")
            bathrooms = gr.Slider(1, 3, value=2, step=1, 
                                label="Nombre de salles de bain")
            age_years = gr.Slider(0, 50, value=10, step=1, 
                                label="Âge de la maison (années)")
        
        with gr.Column():
            lot_size = gr.Slider(2000, 10000, value=5000, step=500, 
                               label="Taille du terrain (pieds carrés)")
            garage_spaces = gr.Slider(0, 2, value=1, step=1, 
                                    label="Places de garage")
            neighborhood_score = gr.Slider(1, 10, value=5, step=1, 
                                         label="Score du quartier")
            predict_btn = gr.Button(" Prédire le Prix", variant="primary")
    
    result_text = gr.Markdown()
    
    with gr.Row():
        feature_importance = gr.Image(label="Importance des Caractéristiques")
        input_summary = gr.Image(label="Résumé des Inputs")
    
    prediction_viz = gr.Image(label="Prédiction avec Intervalle de Confiance")
    
    predict_btn.click(
        fn=predict_price,
        inputs=[square_feet, bedrooms, bathrooms, age_years,
               lot_size, garage_spaces, neighborhood_score],
        outputs=[result_text, feature_importance, input_summary, prediction_viz]
    )

if __name__ == "__main__":
    demo.launch()