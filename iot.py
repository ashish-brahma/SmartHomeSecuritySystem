#! /usr/bin/python

# Imports
import RPi.GPIO as GPIO
import time
import requests

# Set the GPIO naming convention
GPIO.setmode(GPIO.BCM)

# Turn off GPIO warnings
GPIO.setwarnings(False)

# Set a variable to hold the GPIO Pin identity
pinpir = 24
pinmag = 23
pinvib = 9

# Set GPIO pin as input
GPIO.setup(pinpir, GPIO.IN)
GPIO.setup(pinmag, GPIO.IN)
GPIO.setup(pinvib, GPIO.IN)

# Variables to hold the current and last states
currentstate = [0,0,0]
previousstate = [0,0,0]

try:
	print("Waiting for sensors to settle ...")
	
	# Loop until PIR output is 0
	while GPIO.input(pinpir) == 1 or GPIO.input(pinmag) == 1 or GPIO.input(pinvib) == 1:
	
		currentstate[0] = 0
		currentstate[1] = 0
		currentstate[2] = 0

	print("    Ready")
	
	# Loop until users quits with CTRL-C
	while True:
	
		# Read PIR state
		currentstate[0] = GPIO.input(pinpir)  
		currentstate[1] = GPIO.input(pinmag)
		currentstate[2] = GPIO.input(pinvib)

		# If the PIR is triggered
		if currentstate[0] == 1 and previousstate[0] == 0:
			print("PIR Detected")
			
			# Your IFTTT URL with event name, key and json parameters (values)
			r = requests.post('https://maker.ifttt.com/trigger/sens_1/with/key/nwKa2db3VvfkdHE8kMhbU47RJkYkPmXXOqDKHHaF1xm', params={"value1":"none","value2":"none","value3":"none"})
			
			# Record new previous state
			previousstate[0] = 1
			
			time.sleep(1)
			
		# If the PIR has returned to ready state
		elif currentstate[0] == 0 and previousstate[0] == 1:
		
			print("Ready")
			previousstate[0] = 0
            
        
		if currentstate[1] == 1 and previousstate[1] == 0:
			print("Door Magnet detached")
			
			# Your IFTTT URL with event name, key and json parameters (values)
			r = requests.post('https://maker.ifttt.com/trigger/sens_2/with/key/nwKa2db3VvfkdHE8kMhbU47RJkYkPmXXOqDKHHaF1xm', params={"value1":"none","value2":"none","value3":"none"})
			
			# Record new previous state
			previousstate[1] = 1
			
			time.sleep(1)
			
		elif currentstate[1] == 0 and previousstate[1] == 1:
		
			print("Ready")
			previousstate[1] = 0
			
		if currentstate[2] == 1 and previousstate[2] == 0:
			print("Vibration detected")
			
			# Your IFTTT URL with event name, key and json parameters (values)
			r = requests.post('https://maker.ifttt.com/trigger/1234/with/key/nwKa2db3VvfkdHE8kMhbU47RJkYkPmXXOqDKHHaF1xm', params={"value1":"none","value2":"none","value3":"none"})
			
			# Record new previous state
			previousstate[2] = 1
			
			time.sleep(5)
			
		elif currentstate[2] == 0 and previousstate[2] == 1:
		
			print("Ready")
			previousstate[2] = 0

		# Wait for 10 milliseconds
		time.sleep(0.01)

except KeyboardInterrupt:
	print("    Quit")

	# Reset GPIO settings
	GPIO.cleanup()

