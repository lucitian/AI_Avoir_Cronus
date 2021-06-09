import plotly.offline as opy
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os, pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from plotly.subplots import make_subplots
import plotly.express as px
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Our Classes
from .helpers import covid_model
from .helpers import LogisticRegression
from .Generate import uwu

sys.modules['LogisticRegression'] = LogisticRegression

converter = {
    'yes': 0,
    'no': 1,
    'na': 2,
    'female': 0,
    'male': 1,
    'outpatient': 0,
    'inpatient': 1
}

covid1 = None
intubate1 = None 
icu1 = None

# Create your views here.
def home(request):
    global covid1 
    global intubate1
    global icu1
    state = {
        'covid': None,
        'intubate': None,
        'icu': None,
        'error': None
    }

    if request.POST: 
        age = int(request.POST['age'])
        sex = request.POST['sex'].lower()
        p_type = request.POST['patient_type'].lower()
        pneumonia = request.POST['pneumonia'].lower()
        pregnant = request.POST['pregnancy'].lower()
        diabetes = request.POST['diabetes'].lower()
        copd = request.POST['copd'].lower()
        asthma = request.POST['asthma'].lower()
        imsupr = request.POST['imsupr'].lower()
        hypertension = request.POST['hypertension'].lower()
        other_diseases = request.POST['other_diseases'].lower()
        cardiovascular = request.POST['cardiovascular'].lower()
        obesity = request.POST['obesity'].lower()
        renal_chronic = request.POST['renal_chronic'].lower()
        tobacco = request.POST['tobacco'].lower()
        other_contact = request.POST['other_contact'].lower()

        model = None
        df = None
        try:
            #model = open(".model.sav")
            module_dir = os.path.dirname(__file__)  # get current directory
            file_path = os.path.join(module_dir, 'model.sav') 
            model = pickle.load(open(file_path, 'rb'))
            df = pd.read_csv(open(os.path.join(module_dir, 'covid.csv'), 'rb'))[100000:230000]
            df = df[df['covid_res'] != 2]
            df = df.drop(['id', 'entry_date', 'date_symptoms', 'date_died'], axis = 1)
        except IOError as e:
            print("File not accessible")
            state['error'] = e
            print(state['error'])
        
        if not state['error']:
            feats = [
                age, converter[sex], converter[p_type], converter[pneumonia], converter[pregnant], converter[diabetes],
                converter[copd], converter[asthma], converter[imsupr], converter[hypertension], converter[other_diseases],
                converter[cardiovascular], converter[obesity], converter[renal_chronic], converter[tobacco], converter[other_contact]
            ]   

            predicted = model.predict(feats)
            #print(predicted)
            predicted = 1 if predicted > 0.5 else 0

            mod_clf = DecisionTreeClassifier()

            X_icu = df.drop(['covid_res', 'icu', 'intubed'], axis=1)
            X_intubate = df.drop(['covid_res', 'intubed', 'icu'], axis=1)

            y_icu = df['icu']
            y_intubate = df['intubed']

            X_train, X_test, y_train, y_test = train_test_split(X_icu, y_icu, test_size=0.5, random_state=2)
            X_train1, X_test1, y_train1, y_test1 = train_test_split(X_intubate, y_intubate, test_size=0.5, random_state=2)

            clf = mod_clf.fit(X_train,y_train)
            clf1 = mod_clf.fit(X_train1,y_train1)

            y_pred = clf.predict([feats])
            y_pred1 = clf1.predict([feats])

            print(y_pred, y_pred1, predicted)
            state['covid'] = predicted 
            state['intubate'] = y_pred1[0]
            state['icu'] = y_pred[0]
            covid1 = state['covid']
            intubate1 = state['intubate'] 
            icu1= state['icu']

        else:
            print("may error ldos haha")
    elif request.GET:
        print(covid1, intubate1, icu1)
        uwu(covid1, intubate1, icu1)

    return render(request, 'directories/first_pane.html', state)

def simulation(request):
    context = {
        'graph': None,
        'transmission': "",
        'recovery': "", 
        'pop': "", 
        'inf': "", 
        'recovered': "",
        'dt': "", 
        'dur': "",
        'error': None 
    }

    fig = make_subplots(
            rows=2, 
            cols=3, 
            subplot_titles=(
                'Susceptible', 
                'Infected',
                'Recovered', 
                'Δ Susceptibles',
                'Δ Infections', 
                'Δ Recovereds'
            ),
        )

    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name="Susceptible"),row=1, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name="Infected"),row=1, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name="Recovered"),row=1, col=3)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name="Rate of Change of Susceptibles"),row=2, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name="Infection Rate"),row=2, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name="Recovery Rate"),row=2, col=3)

    fig.update_layout(
        height=700, width=1090 , title_text="SIR Epidemic Model",
    )
    if request.POST:
        context['transmission'] = request.POST['transmission_rate']
        context['recovery'] = request.POST['recovery_rate']
        context['pop'] = request.POST['population']
        context['inf'] = request.POST['infected']
        context['recovered'] = request.POST['recovered']
        context['dt'] = request.POST['time_step']
        context['dur'] = request.POST['duration']

        m = covid_model.SIRModel(
            float(context['transmission']), 
            float(context['recovery']), 
            float(context['pop']), 
            int(context['inf']), 
            int(context['recovered']), 
            float(context['dt'])
        )

        fig.update_layout(title_text=f"SIR Epidemic Model R0 = {m.R0()}")

        time_frame = m.run(int(context['dur']))

        df = pd.DataFrame({
            'time_frame': time_frame, 
            'S': m.S_history, 
            'I': m.I_history, 
            'R': m.R_history, 
            'dS': m.daily_susceptibility_rate,
            'dI': m.daily_infection_rate,
            'dR': m.daily_recovery_rate
        })

        Frame_1 = []
        times = []
        S = []
        I = []
        R = []
        dS = []
        dI = []
        dR = []
        for i, row in df.iterrows():
            times.append(row['time_frame'])
            S.append(row['S'])
            I.append(row['I'])
            R.append(row['R'])
            dS.append(row['dS'])
            dI.append(row['dI'])
            dR.append(row['dR'])
            Frame_1.append(go.Frame(data=[
                go.Scatter(x=times, y=S, mode='lines'),
                go.Scatter(x=times, y=I, mode='lines'),
                go.Scatter(x=times, y=R, mode='lines'),
                go.Scatter(x=times, y=dS, mode='lines'),
                go.Scatter(x=times, y=dI, mode='lines'),
                go.Scatter(x=times, y=dR, mode='lines')
            ], traces = [0,1,2,3,4,5]))
        
        fig.frames = Frame_1

        button = dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=50, redraw=False), 
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode='immediate')])
        button1 = {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                'label': 'Pause',
                    'method': 'animate'
                }
        button2 = {
                    'args': [[None], {'frame': {'duration': 25, 'redraw': False}, 'mode': 'immediate', 'fromcurrent' : False,
                    'transition': {'duration': 0}}],
                'label': 'Replay',
                    'method': 'animate',
                }
        fig.update_layout(updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=0,
                                    x=1.05,
                                    xanchor='left',
                                    yanchor='bottom',
                                    buttons=[button, button1, button2] )
                                            ],
                        xaxis=dict(range=[0, int(context['dur'])], autorange=False),
                        yaxis=dict(range=[0, int(context['pop'])], autorange=False),
                        transition = {'duration': 25})
        fig.update_layout(
            xaxis2_range=[0, int(context['dur'])], xaxis2_autorange=False, yaxis2_range=[0, int(context['pop'])], yaxis2_autorange=False,
            xaxis3_range=[0, int(context['dur'])], xaxis3_autorange=False, yaxis3_range=[0, int(context['pop'])], yaxis3_autorange=False,
            xaxis4_range=[0, int(context['dur'])], xaxis4_autorange=False, yaxis4_range=[0, float(max(dS)) + (0.1 * float(max(dS)))], yaxis4_autorange=False,
            xaxis5_range=[0, int(context['dur'])], xaxis5_autorange=False, yaxis5_range=[min(dI), float(max(dS)) + (0.1 * float(max(dS)))], yaxis5_autorange=False,
            xaxis6_range=[0, int(context['dur'])], xaxis6_autorange=False, yaxis6_range=[0, float(max(dS)) + (0.1 * float(max(dS)))], yaxis6_autorange=False,
        )

        context['graph'] = opy.plot(fig, auto_open=False, output_type='div', auto_play=False)
    else:
        context['graph'] = opy.plot(fig, auto_open=False, output_type='div', auto_play=False)


    return render(request, 'directories/second_pane.html', context)

def database(request):
    context = {
        'data' : []
    }

    if request.POST:
        f = request.FILES['myfile']
        fs = FileSystemStorage()
        module_dir = os.path.dirname(__file__)  
        file_path = os.path.join(module_dir, f.name) 
        filename = fs.save(file_path, f)
        
        """
        df = pd.read_csv(open(file_path, 'rb'))[0:20]

        columns = []
        counter = 0
        for i, row in df.iterrows():
            columns.append([row['id'],row['sex'],row['patient_type'],row['entry_date'],row['date_symptoms'],row['date_died'],row['intubed'],row['pneumonia'],row['age'],
            row['pregnancy'], row['diabetes'], row['copd'], row['asthma'], row['inmsupr'], row['hypertension'], row['other_disease'], row['cardiovascular'],row['obesity'],
            row['renal_chronic'],row['tobacco'],row['contact_other_covid'],row['covid_res'],row['icu']])
        """
        module_dir = os.path.dirname(__file__)
        file_path = os.path.join(module_dir, 'model.sav') 
        model = pickle.load(open(file_path, 'rb'))

        data = model.X_test[300:400]
        data = np.hstack((data,model.y_test[300:400]))

        columns = []
        for i in data:
            # columns.append([row['sex'],row['patient_type'],row['intubed'],row['pneumonia'],row['age'],
            # row['pregnancy'], row['diabetes'], row['copd'], row['asthma'], row['inmsupr'], row['hypertension'], row['other_disease'], row['cardiovascular'],row['obesity'],
            # row['renal_chronic'],row['tobacco'],row['contact_other_covid'],row['covid_res'],row['icu']])
            columns.append([i.item((0,0)), 0, i.item((0,1)),i.item((0,2)),i.item((0,3)),i.item((0,4)),i.item((0,5)),i.item((0,6)),i.item((0,7)),i.item((0,8)),
            i.item((0,9)), i.item((0,10)), i.item((0,11)),i.item((0,12)),i.item((0,13)),i.item((0,14)),i.item((0,15)),i.item((0,16))
            ])
        
        fig = px.line(x=range(len(model.cost_history)), y=np.flip(np.abs(model.cost_history)), title='Cost per Iteration')

        fig.update_layout(
                   xaxis_title='Iterations',
                   yaxis_title='Cost (θ)')

        graph = opy.plot(fig, auto_open=False, output_type='div', auto_play=False)

        context = {
            'data' : columns,
            'graph': graph
        }
            
    return render(request, 'directories/third_pane.html', context)