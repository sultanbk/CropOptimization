import smtplib

server = smtplib.SMTP('smtp.googlemail.com', 587)
server.starttls()
server.login('sbkhelpdesk258@gmail.com', 'imiy lpgt eeum vikw')

print("SMTP server connection successful.")

server.quit()
