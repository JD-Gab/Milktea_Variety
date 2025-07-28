from flask import Flask, render_template, redirect, url_for, request
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

app = Flask(__name__)


data = pandas.read_csv('milktearecommendation4.csv',encoding = 'cp1252')
data2 = pandas.read_csv('milktearecommendation6.csv',encoding = 'cp1252')

milkteaset = {'Black Milk Tea':'blackmilktea.jpg',
              'Taro Milk Tea':'taromilktea.png',
              'Thai Milk Tea':'thaimilktea.png',
              'Brown Sugar Milk Tea':'brownsugarmilktea.jpg',
              'Matcha Milk Tea':'matchamilktea.png',
              'Honeydew Milk Tea':'honeydew.jpg',
              'Strawberry Milk Tea':'strawberrymilktea.png',
              'Mango Milk Tea':'mango.jpg',
              'Coconut Milk Tea':'coconut.jpg',
              'Hokkaido Milk Tea':'hokkaido.jpg',
              'Okinawa Milk Tea':'okinawa.jpg',
              'Mocha Milk Tea':'mocha.jpg',
              'Chocolate Milk Tea':'choco.jpg',
              'Lychee Milk Tea':'lychee.jpg',
              'Jasmine Milk Tea':'jasmine.jpg'}
milkteatype = {1:'Black Milk Tea',2:'Taro Milk Tea',3:'Thai Milk Tea',4:'Brown Sugar Milk Tea',5:'Matcha Milk Tea',6:'Honeydew Milk Tea',7:'Strawberry Milk Tea',8:'Mango Milk Tea',9:'Coconut Milk Tea',10:'Hokkaido Milk Tea',11:'',12:'',13:'',14:'',15:''}

X = data.drop(columns=['Recommended Milk Tea','description'])
y = data['Recommended Milk Tea']
z = data.drop(data.columns[[0,1,2,3,4]], axis=1)

X2 = data2.drop(columns=['Recommended Milk Tea','description'])
y2 = data2['Recommended Milk Tea']
z2 = data2.drop(data2.columns[[0,1,2,3,4]], axis=1)

print(X.values[0][0])
model = DecisionTreeClassifier()
knnmodel = KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(X2.values,y2)
model.fit(X.values,y)

@app.route('/')
def index():

    return render_template('index.html')


@app.route('/recommend')
def recommend():
    return render_template('search.html')


@app.route('/recoprocessing',methods=['POST'])
def recoprocessing():
    if request.method == 'POST':
        sugar = request.form['sugar']
        milk = request.form['milk']
        tapioca = request.form['tapioca']
        greentea = request.form['greentea']
        coffee = request.form['coffee']

        return redirect(url_for('result', sugar = sugar, milk = milk, tapioca = tapioca, greentea = greentea, coffee = coffee))

    return redirect(url_for('recommend'))

@app.route('/result')
def result():
    a = request.args.get('sugar')
    b = request.args.get('milk')
    c = request.args.get('tapioca')
    d = request.args.get('greentea')
    e = request.args.get('coffee')
    description = 'x'
    image = ''

    
    datalist = [a,b,c,d,e]
    datalist2 = [int(a) for a in datalist]
    
    predicted_value = model.predict([datalist2])
    knn_predicted = knnmodel.predict([datalist2])[0]
    milktea2 = data.iloc[knn_predicted][5]

    string_pv = np.array(predicted_value[0]).tolist()

    for i in z.values[0:15]:
        milktea = np.array(i[0]).tolist()
        if string_pv == milktea:
            description = np.array(i[1]).tolist()
            image = milkteaset[string_pv]
        if milktea2 == milktea:
            description2 = np.array(i[1]).tolist()
            image2 = milkteaset[milktea2]

    return render_template('blog.html', milktea = predicted_value[0], description = description, image = image, milktea2 = milktea2, description2 = description2, image2 = image2)


if __name__ == '__main__':
    app.run(debug=True)
