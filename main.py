from flask import Flask , render_template , request
app = Flask(__name__)
import pickle


#open a file where u stored the pickle data
file= open('model.pkl', 'rb')
clf= pickle.load(file)
file.close()




@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        mydict = request.form
        fever = int(mydict['fever'])
        age = int(mydict['age'])
        pain = int(mydict['pain'])
        runnynose = int(mydict['runnynose'])
        diffbreath = int(mydict['diffbreath'])
        #code for inference
        inputFeatures = [fever, age, pain, runnynose, diffbreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template ('show.html', inf = round(infProb * 100))
    return render_template ('index.html')

    #return  str(infprob)


    if __name__=='__main__':
        app.run(debug=True)
