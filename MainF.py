import streamlit as st

import base64
import pandas as pd
import streamlit as st
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import base64


# =================

# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')   
 


st.title("PREDICTING HEALTHCARE RECORDS")

typee = st.selectbox( 'CHOOSE HERE ',('DEFAULT','ADMIN', 'USER'))

if typee=="DEFAULT":
    # st.text("Welcome to our page !!!")

    st.markdown(f'<h1 style="color:#006400;font-size:24px;">{" Welcome to our Page !!! "}</h1>', unsafe_allow_html=True)

if typee=="ADMIN":
    
    st.success("Wlecome Admin!!!")
    
    
    adname=st.text_input("Enter Name","Name")
    passs=st.text_input("Enter Password","Password",type="password")
    aa=st.button("Submit")
    
    # if aa:
        
    if adname=="Admin" and passs=="12345":
        st.success("Admin Login Successfully !!!")
        
        username=st.text_input("Enter Patient Name")
        
        file_up = st.file_uploader("View Data", type=["txt"])
        
        if file_up==None:
            st.text("Please Upload Files")
            
        else:
            
            nn=username + '.txt'
            dataframe=pd.read_csv(nn)
        
            st.text("View Details")
            st.text(dataframe)
            
            
            
            # with open(os.path.join("upload",file_up.name),"wb") as f:
            #   f.write(file_up.getbuffer())
            
            
            st.success("View Successfully !!!")        
        
if typee=="USER":
    st.success("Wlecome User!!!")    
    
    
    # ============= REGISTERATION AND LOGIN ==================
    
    import pandas as pd
    
    # df = pd.read_csv('login_record.csv')
    
    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    
    col1, col2 = st.columns(2)
    
    
        
    with col1:
    
        UR1 = st.text_input("Login User Name",key="username")
        psslog = st.text_input("Password",key="password",type="password")
        # tokenn=st.text_input("Enter Access Key",key="Access")
        agree = st.checkbox('LOGIN')
        
        if agree:
            try:
                
                df = pd.read_csv(UR1+'.csv')
                U_P1 = df['User'][0]
                U_P2 = df['Password'][0]
                if str(UR1) == str(U_P1) and str(psslog) == str(U_P2):
                    st.write('Successfully Login !!!')    
                    st.write('\n')
                    st.write('\n')
                    st.write('\n')
                    st.write('\n')
                    st.write('\n')        
                    
                    
                    df=pd.read_csv("Data.csv")    
                    
                    print("--------------------------------")
                    print("Data Selection")
                    print("--------------------------------")
                    print(df.head(15))
                    
                    print("--------------------------------")
                    print("Handling Missing Values")
                    print("--------------------------------")                    
                    
                    print(df.isnull().sum())
                    
                    print("--------------------------------")
                    print("Before Label Encoding")
                    print("--------------------------------")   
                    
                    print(df['gender'].head(15))
                    
                    print("--------------------------------")
                    print("After Label Encoding")
                    print("--------------------------------")                       
                    
                    
                    label_encoder = preprocessing.LabelEncoder() 

                    df['gender']=label_encoder.fit_transform(df['gender'])                     

                    df['gluc']=label_encoder.fit_transform(df['gluc'])    
                    
                    df['cholesterol']=label_encoder.fit_transform(df['cholesterol'])  
                    
                    # df['Location']=label_encoder.fit_transform(df['Location'])  

                    # df['Type']=label_encoder.fit_transform(df['Type'])  
                    
                    print(df['gender'].head(15))
                    
                    # drop columns
                    
                    df=df.drop(['Author'],axis=1)
                    
                    
                    X = df.drop("BloodCancer",axis=1)
                    Y = df["BloodCancer"]
                    
                    print("----------------------------------------")
                    print("DATA SPLITTING")
                    print("------------------------------------")
                    print()
                    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
                    
                    
                    print()
                    print("Total Number Of data      = ", len(X))
                    print()
                    print("Total Number Of Test data = ", len(x_test))
                    print()
                    print("Total Number Of Train data = ", len(x_train))
                    print()                    
                    
                    
                     # === RANDOM FOREST =====
                    
                    from sklearn.ensemble import RandomForestClassifier
                    
                    regressor = RandomForestClassifier(n_estimators = 10) 
                     
                    # fit the regressor with x and y data
                    regressor.fit(x_train, y_train) 
                    
                    Y_pred = regressor.predict(x_train)
                    
                    from sklearn import metrics
                    
                    Accuracy_rf=metrics.accuracy_score(y_train,Y_pred)*100
                     
                     
                    print("----------------------------------------")
                    print("RANDOM FORES --> RF")
                    print("------------------------------------")
                    print()
                    print("1. Accuracy =",Accuracy_rf )
                    print()
                    print(metrics.classification_report(y_train,Y_pred))                    
                    
                    st.text("Please Enter Below Details")
                    
                    idd=st.text_input("Enter User ID ",'0')
    
                    age_days=st.text_input("Enter Age Days ",'0')
    # diabe=int(diabe)
    
                    age_year=st.text_input("Enter Age in Years ",'0')
    # bp=int(bp)
    
                    gender=st.text_input("Enter Gender ",'0')
    # chol=int(chol)
    
                    height=st.text_input("Enter Height ",'0')
                    
                    
                    weight=st.text_input("Enter Weight ",'0')
    # bmi=int(bmi)
    
                    ap_hi=st.text_input("Enter ap_hi ",'0')
    # smoker=int(smoker)
    
                    ap_lo=st.text_input("Enter ap_lo ",'0')
                    # stroke=int(stroke)
                    
                    cholesterol=st.text_input("Enter cholesterol ",'0')
                    # heart=int(heart)
                    
                    gluc=st.text_input("Enter gluc ",'0')
                    # age=int(age)
                    
                    smoke=st.text_input("Enter smoke ",'0')
                    
                    alco=st.text_input("Enter alco ",'0')
                    
                    active=st.text_input("Enter Sex ",'0')
                    
                    
                    
            
                    Data_reg=[idd,age_days,age_year,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active]
            
                    with open(str(UR1)+'.txt', 'w+') as f:
                         
                        # write elements of list
                        for items in Data_reg:
                            f.write('%s\n' %items)



                    import numpy as np 
                    
                    input_1 = np.array([idd,age_days,age_year,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active]).reshape(1, -1)
                    
                    predicted_data = regressor.predict(input_1)
                    
    
                    st.text("PREDICTION")
                
                    if predicted_data==1:
                        print("----------------------------")
                        print()
                        print("Health is POOR")
                        pred="Health is POOR"
                        st.text("Health is POOR")

               
                    else:
                        print("----------------------------")
                        print()
                        print("Health is GOOD")  
                        pred="Health is GOOD"
                
                        st.text("Health is GOOD")

                    import hashlib
                    from datetime import datetime
                    class Block:
                    
                        def __init__(self,  data, previous_hash):
                          self.timestamp =  datetime.now()
                          self.data = pred
                          self.previous_hash = previous_hash
                          self.hash = self.calc_hash()
                          self.next = None
                    
                    
                        def calc_hash(self):
                            hash_str = "We are going to encode this string of data!".encode('utf-8')
                            return hashlib.sha256(hash_str).hexdigest()
                    
                    class Blockchain:
                    
                        def __init__(self):
                            self.head = None
                            self.next = None
                    
                        def add_block(self, data):
                    
                            if self.head == None:
                                self.head = Block(data,0)
                    
                            else:
                                current = self.head
                    
                              # loop to the last node of the linkedlist
                                while current.next:
                                    current = current.next
                    
                                # stores the previous has for the next block
                                previous_hash = current.hash
                                current.next = Block(data, previous_hash)
                    
                          
                    
                        def print_blockchain(self):
                    
                            if self.head == None:
                                print("The blockchain is empty")
                                return
                    
                            else:
                                current = self.head
                                while current:
                                    print("Timestamp:", current.timestamp)
                                    print("Data:", current.data)
                                    print("Hash:", current.hash)
                                    print("Previous hash:", current.previous_hash)
                                    print("--------------->")
                    
                                    current = current.next
                    
                    bitcoin = Blockchain()
                    
                    bitcoin.add_block("block 1")
                    bitcoin.add_block("block 2")
                    bitcoin.add_block("block 3")
                    bitcoin.print_blockchain()
                    print(bitcoin.print_blockchain())
                    st.write(bitcoin.print_blockchain())    
                    st.success("Data Uploaded Successfully !!!")                               
                            
                else:
                    st.write('Login Failed!!!')
            except:
                st.write('Login Failed!!!')                    
                    
    
    with col2:
        UR = st.text_input("Register User Name",key="username1")
        pss1 = st.text_input("First Password",key="password1",type="password")
        pss2 = st.text_input("Confirm Password",key="password2",type="password")
        # temp_user=[]
            
        # temp_user.append(UR)
        
        if pss1 == pss2 and len(str(pss1)) > 2:
            import pandas as pd
            
      
            import csv 
            
            # field names 
            fields = ['User', 'Password'] 
            

            
            # st.text(temp_user)
            old_row = [[UR,pss1]]
            
            # writing to csv file 
            with open(UR+'.csv', 'w') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                    
                # writing the fields 
                csvwriter.writerow(fields) 
                    
                # writing the data rows 
                csvwriter.writerows(old_row)
            st.write('Successfully Registered !!!')
        else:
            
            st.write('Registeration Failed !!!')    
                   
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    