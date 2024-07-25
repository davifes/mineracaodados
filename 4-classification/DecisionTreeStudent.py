import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def main():
    input_file = 'student-por.csv'
    names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
             'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 
             'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
             'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'situation']
    
    features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 
                'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3'] 

    target = 'situation'
    data = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names, delimiter=';') # Nome das colunas                      
    
    # Select features and target
    X = data[features]
    y = data[target]
    
    # Split the data - 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Train the model
    clf = DecisionTreeClassifier(max_leaf_nodes=6)
    clf.fit(X_train, y_train)

    # Plot the decision tree
    plt.figure(figsize=(12,8))
    tree.plot_tree(clf, feature_names=features, class_names=clf.classes_, filled=True)
    plt.show()

if __name__ == "__main__":
    main()
