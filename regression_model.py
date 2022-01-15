import pandas as pd
import streamlit as st
import numpy as np
import pickle
from PIL import Image
import plotly.express as px
import os
import time
from joblib import load






# html_temp = """
# 		<div style="text-align: center"> your-text-here 
# 		</div>
# 		PAGE_CONFIG = {"page_title":"Solar Fx","page_icon":img,"layout":"centered"}
# 		st.set_page_config(**PAGE_CONFIG)	
# 		"""
# PAGE_CONFIG = {"layout":"wide"}
# st.set_page_config(**PAGE_CONFIG)	

def main():
	menu = ['Sales Prediction','Analysis']
	choice = st.sidebar.selectbox('Menu',menu)
	
	if choice == 'Sales Prediction':
		cola, colb, colc = st.columns([1.8,4,1])
		with cola:
			st.empty()
		with colb:
			st.title('Sales Prediction')
		with colc:
			st.empty()
		st.write('')
		st.write('')
		st.write('')	


		
		col1,col2, col3= st.columns(3)
	
		with col1:
			tv = st.number_input("TV promotion budget ($ mn)")
		# with col2:
		# 	radio = st.number_input("Radio promotion budget ($ mn)")
		
		# col3,col4 = st.columns(2) 	
		with col2:
			radio = st.number_input("Radio promotion budget ($ mn)")
		
		with col3:
			sm = st.number_input("Social Media promotion budget ($ mn)")
		

		col4, col5, col6 = st.columns(3)

		with col4:
			inf = st.selectbox('Select Influencer',["Nano","Micro","Macro","Mega"])
				
		inf_dict = {"Nano":1,"Micro":2,"Macro":3,"Mega":4}
		
		with col5:
			
			ch_model = st.selectbox('Select Model',['Random Forest','Lasso Regression','Linear Regression'])
			if ch_model == 'Random Forest':
				loaded_model = pickle.load(open('/home/akash/Desktop/Pers/Regression_Project/regression/finalized_model_Random-Forest_Full_Data.sav', 'rb'))

			elif ch_model=='Lasso Regression':
				loaded_model = pickle.load(open('/home/akash/Desktop/Pers/Regression_Project/regression/finalized_model_Lasso_Reg_Full_Data.sav', 'rb'))

			else:
				loaded_model = pickle.load(open('/home/akash/Desktop/Pers/Regression_Project/regression/finalized_model_Linear_Reg_Full_Data.sav', 'rb'))

		with col6:
			st.empty()
			pass
		
		st.write('')
		st.write('')
		if st.button('Predict',key=2):
			# st.write(inf)
			inf = inf_dict.get(inf)
			if inf == 1:
				single_value = np.array([tv,radio,sm,1,0,0]).reshape(1,-1)
			elif inf == 2:
				single_value = np.array([tv,radio,sm,0,1,0]).reshape(1,-1)
			elif inf == 3:
				single_value = np.array([tv,radio,sm,0,0,1]).reshape(1,-1)
			else:
				single_value = np.array([tv,radio,sm,0,0,0]).reshape(1,-1)	
			
			# st.write(inf)
			# st.write(np.array([tv,radio,sm,inf]))
			# st.write(np.array([tv,radio,sm,inf]).reshape(1,-1))
			#single_value=np.array(tv).reshape(1,-1)
			#single_value = np.array([tv,sm,inf]).reshape(1,-1)
			sc = load('/home/akash/Desktop/Pers/Regression_Project/regression/std_scaler.bin')
			single_value = sc.transform(single_value)
			
			prediction_single = loaded_model.predict(single_value)
			
			with st.spinner('Predicting the results!...'):
				time.sleep(0.3)
				st.success('Result is ready!')				
				st.write('Sales Prediction: $ {} mn '.format(np.round(int(prediction_single[0]))))

	else:
		col11,col31,col41= st.columns([3,4,1])
		
		with col11:
			st.empty()
		with col31:
			st.title('Analysis')
		with col41:
			st.empty()
		st.write('')
		st.write('')
		
		with st.expander('Plots'):
			
			#if choice2 == "Major Terms/Dates":
			menu3 = ['Pair Plot','Distribution Plot','Correlation Matrix']
			#choice2 = st.sidebar.selectbox('Select model',menu2)
			
			choice3 = st.selectbox('Plot Type',menu3)
			if choice3==menu3[0]:
				img2 = Image.open("/home/akash/Desktop/Pers/Regression_Project/regression/pairplot.png")
				st.image(img2,width = 120,use_column_width=True)

				
			elif choice3==menu3[1]:
				img1 = Image.open("/home/akash/Desktop/Pers/Regression_Project/regression/Distribution.png")
				st.image(img1,width = 120,use_column_width=True)
			else:
				img3 = Image.open("/home/akash/Desktop/Pers/Regression_Project/regression/Correlation-Matrix.png")
				st.image(img3,width = 120,use_column_width=True)
		st.write('')
		st.write('')
		st.write('')
		st.write('')
			
		


		with st.expander('Model Coefficients'):
			#st.subheader('Feature Importance in Predicting Sales')
			st.write('')
			st.write('')
			
			menu2 = ['Linear Regression','Lasso Regression','Ridge Regression','Random Forest']
			#choice2 	= st.sidebar.selectbox('Select model',menu2)
			choice2 = st.selectbox('Algorithm',menu2)

			if choice2==menu2[0]:
				## Linear Regression
				st.subheader('Linear Regression')
				chart_data = pd.DataFrame([[3.54702454 , 0.16783844 , -0.02387392]],columns=["TV","Social Media",'Influencer'])
				chart_data = chart_data.T
				st.bar_chart(chart_data)
				st.write('')
				#st.caption(' change in the Sales for one unit of change in the predictor variable while holding other predictors in the model constant. ')

			elif choice2==menu2[1]:
				## For Lasso Regression
				st.subheader('Lasso Regression')
				chart_data2 = pd.DataFrame([[3.55147831 , 0. ,-0. ]],columns=["TV","Social Media",'Influencer'])
				chart_data2 = chart_data2.T
				st.bar_chart(chart_data2)
				st.write('')


			elif choice2==menu2[2]:
			## For Ridge Regression
				st.subheader('Ridge Regression')
				chart_data3 = pd.DataFrame([[3.54352364 , 0.21042874 ,-0.07284456]],columns=["TV","Social Media",'Influencer'])
				chart_data3 = chart_data3.T
				st.bar_chart(chart_data3)
				st.write('')
			else:	
			## For Random Forest Regression
				st.subheader('Random Forest Regression')
				chart_data4 = pd.DataFrame([[0.99471601 , 0.00468622 ,0.00059778]],columns=["TV","Social Media",'Influencer'])
				chart_data4 = chart_data4.T
				st.bar_chart(chart_data4)
				st.write('')

			

if __name__=='__main__':
	main()	