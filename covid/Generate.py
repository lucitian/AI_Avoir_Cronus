#Python Libraries
import datetime
from fpdf import FPDF
pdf = FPDF(orientation='P', unit='in', format='letter')
name = "Microbic Baja"

#sizes in inches
width = 8.5
height = 11
spacing = .3

'''One Page Report'''
def title(day, pdf):
    pdf.set_font('Arial', 'B', 16)
    pdf.write(1.5,f"Decision Support System Report")
    pdf.set_x(.5)
    pdf.set_font('Helvetica', '', 12)
    pdf.write(2,f"{day}")

def format(x,y):
    pdf.set_font('Times', '', 12)
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.set_xy(x,y)

def alert_format(x,y):
    pdf.set_font('Times', '', 12)
    pdf.set_text_color(r=255, g=0, b=0)
    pdf.set_xy(x,y)


def generate_report(day, icu_val, intube_val, covid_val):
    #Title:
    pdf.add_page()
    title(day, pdf)

    #Content - Plot/Graphs
    #image 1 
    text = ""
    pdf.image("covid/static/directories/images/icu_icon.png", 1.5, 2, 1.5)
    if icu_val == 1:
        format(4.5,2.25)
        text = "The patient might need to be admitted into the Intensive Care Unit. Perform additional medical procedures to validate."
    elif icu_val == 0:
        alert_format(4.5,2.25)
        text = "The patient is not recommended for Intensive Care Unit admission. Perform additional medical check ups to validate."
    pdf.multi_cell(3,.3,text, 0, 0 , 'J')

    #image 2
    text = ""
    pdf.image("covid/static/directories/images/intube_icon.png", 1.5, 4.5, 1.5)
    if intube_val == 1:
        format(4.5,4.75)
        text = "The patient might need intubation based on the findings of the Decision Support System. Please perform additional medical checkups for confirmation."
    elif intube_val == 0:
        alert_format(4.5,4.75)
        text = "The patient might not need intubation based on the findings of the Decision Support System. Additional medical checkups are required for confirmation."
    pdf.multi_cell(3,.3,text, 0, 0 , 'J')

    #image 3
    text = ""
    pdf.image("covid/static/directories/images/virus_icon.png", 1.5, 7, 1.5)
    if covid_val == 1:
        format(4.5,7.25)
        text = "The patient might have COVID-19 based on the findings of the Decision Support System. This is just a computer-aided recommendation, please consult a physician if certain symptoms persists."
    elif covid_val == 0:
        alert_format(4.5,7.25)
        text = "The patient might not have COVID-19 based on the findings of the Decision Support System. This is just a computer-aided recommendation, please consult a physician if certain symptoms persists."
    pdf.multi_cell(3,.3,text, 0, 0 , 'J')
    
    #Legend
    format(1,9)
    pdf.multi_cell(6.5, .3, align='J', txt="Legend: \n\t\t\t\t\tRed Text Indicates - Action Needed\n\t\t\t\t\tBlack Text Indicates - Normal")
    
    pdf.output('test.pdf','F') 

def uwu(icu_val, intube_val, covid_val):
    #get date
    day = datetime.datetime.today().strftime ('%d/%m/%Y')
    str_day = str(day)

    generate_report(str_day, icu_val, intube_val, covid_val)

