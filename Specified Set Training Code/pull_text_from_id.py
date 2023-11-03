import requests


"""
USE: Access the raw text from the online database
"""

def pull_text(id_str):
	#dowload link for text stored as [ADDRESS]/FIRSTLETTER/SECONDLETTER/THIRDLETTER/FOURTHLETTER/FULL_ID/FULL_ID.ocr
	url = "https://download.industrydocuments.ucsf.edu/"+id_str[0]+"/"+id_str[1]+"/"+id_str[2]+"/"+id_str[3]+"/"+id_str+"/"+id_str+".ocr"
	r = requests.get(url, allow_redirects=True) #pull raw bytes from site
	#write binary to a text doc to be used by model
	f = open("temp_read.txt", "wb")
	f.seek(0) 
	f.write(r.content)
	f.close()
	return "temp_read.txt" #return file name for use in model


