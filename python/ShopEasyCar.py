import pandas as pd
import pyodbc
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from transformers import pipeline
from tqdm import tqdm



# telechargement du lexicon vader
# nltk.download('vader_lexicon')

# function fetch données dans la table customer_reviews
def get_data_customer_reviews_sql():
  # definition paramètre de connexion localDB_Manu_RMT
  conn_data = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=(localdb)\localDB_Manu_RMT;"
    "Database=ShopEasyCar;"
    "Trusted_Connection=yes;"
  )

  # connexion
  conn = pyodbc.connect(conn_data)

  # SQL query
  query = "SELECT ReviewID, CustomerID, ProductID, ReviewDate, Rating, Comment FROM customer_reviews;"

  # éxécute la requête et stock les données dans un DF
  datas = pd.read_sql(query, conn)

  # ferme la connexion
  conn.close()

  return datas

# lance 1 par 1 sur chaque ligne mais trop lent
# solution lancer par batch de 100 
def get_score_sentences(reviews):
  analyse = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

  res =  analyse(reviews)[0]
 
  return res['label'], res['score'] 
#customer_reviews_df[['LabelModel', 'SentimentScore']] = customer_reviews_df['Comment'].apply(get_score_sentences).apply(pd.Series) 


def label_final(colonne):
    note = colonne["Rating"]
    current_label = colonne["LabelModel"]
    if note == 3:
        return "NEUTRE"
    else:
        return current_label


customer_reviews_df = get_data_customer_reviews_sql()
customer_list = customer_reviews_df["Comment"].astype(str).tolist() # crée une liste des commentaires



# Solution 2 : On lance par lot de 100
batch_size = 500
results = []

#data_models = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
data_models = pipeline("sentiment-analysis", model="tblard/tf-allocine")


for i in tqdm(range(0, len(customer_list), batch_size)):
    batch = customer_list[i:i + batch_size]
    batch_results = data_models(batch)
    results.extend(batch_results)



# Extraire label & score
customer_reviews_df["ConfianceScore"] = [r["score"] for r in results]
customer_reviews_df["LabelModel"] = [r["label"] for r in results]
customer_reviews_df['labelFinal'] =  customer_reviews_df.apply(label_final)
customer_reviews_df.to_csv(r"C:\Users\manra\Documents\Projects\WorkSpace\ShopEasyCar\SentimentScoreFinal.csv",index=False,sep=";",encoding="utf-8-sig",mode="w")


# Compter le nombre de chaque label final
repartition = customer_reviews_df["LabelModel"].value_counts()

# Définir couleurs personnalisées
couleurs = {
    "POSITIVE": "#2ecc71",  # vert
    "NEUTRE": "#f1c40f",    # jaune
    "NEGATIVE": "#e74c3c"   # rouge
}

# Générer le camembert
plt.figure(figsize=(6, 6))
plt.pie(
    repartition,
    labels=repartition.index,
    colors=[couleurs.get(label, "#95a5a6") for label in repartition.index],
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Répartition des sentiments client", fontsize=14)
plt.axis("equal")  # Pour un cercle parfait
plt.tight_layout()
plt.show()




