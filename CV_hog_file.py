import cv2
import numpy as np
import math
import random
from numpy  import array
from numpy import zeros

def hog_descriptor(image_name):
	
	img = cv2.imread(image_name,cv2.IMREAD_COLOR) 
	height, width = img.shape[:2]
	
	#print(img)
	new_img = []
	for i in range(0,height):
		new_img.append([])
	for i in range(0,height):
		for j in range(0,width):
			new_img[i].append(round(0.299*img[i][j][2]+0.587*img[i][j][1]+0.114*img[i][j][0]))
	#print(new_img[0][0])
	a = np.array(new_img, dtype=np.uint8)
	#print(new_img)
	new_height, new_width = a.shape[:2]
	#print(new_height,new_width)


	x_image = []
	for i in range(0,new_height-2):
		x_image.append([])
	for i in range(0,new_height-2):
		for j in range(0,new_width-2):
			#Normalization done below by taking the absolute value and then dividing by 3 (0-755 max value range)
			x_image[i].append((-new_img[i][j]+new_img[i][j+2]-new_img[i+1][j]+new_img[i+1][j+2]-new_img[i+2][j]+new_img[i+2][j+2])/3)
	b = np.array(x_image, dtype=np.uint8)
	
	
	#Gradient along Y-axis to find Horiontal Edges using Prewitt's Gy Kernel
	y_image = []
	for i in range(0,new_height-2):
		y_image.append([])

	for i in range(0,new_height-2):
		for j in range(0,new_width-2):
			#Normalization done below by taking the absolute value and then dividing by 3 (0-755 max value range)
			y_image[i].append((new_img[i][j]+new_img[i][j+1]+new_img[i][j+2]-new_img[i+2][j]-new_img[i+2][j+1]-new_img[i+2][j+2])/3)
	c = np.array(y_image, dtype=np.uint8)

	
	
	#Computation of Magnitude Image followed by normaliation by division by square root of 2. Math.Sqrt function is used below.
	mag_image = []
	for i in range(0,new_height-2):
		mag_image.append([])

	grad_angle = []
	for i in range(0,new_height-2):
		grad_angle.append([])

	for i in range(0,new_height-2):
		for j in range(0,new_width-2):
			mag_image[i].append(np.sqrt((x_image[i][j]*x_image[i][j])+(y_image[i][j]*y_image[i][j]))/np.sqrt(2))
			#Computation of Gradient Angle Matrix for Gx = 0 Conditions
			if(x_image[i][j]== 0):
				if(y_image[i][j]>0):
					grad_angle[i].append(90)
				elif(y_image[i][j]<0):
					grad_angle[i].append(-90+180)
				else:
					grad_angle[i].append(0)
			else:
				#General case computation of Grad Angle. Math.degrees converts radians to degrees while math.arctan does the Tan inverse operation.
				angle = math.degrees(np.arctan((y_image[i][j]/x_image[i][j])))
				if(angle<0):
					grad_angle[i].append(angle+180)
				else:
					grad_angle[i].append(angle)
			#if(grad_angle[i][j]>170):
			#	grad_angle[i][j] = grad_angle[i][j] - 180
			if(grad_angle[i][j] == -0.0):
				grad_angle[i][j] = 0.0
			if(grad_angle[i][j] > 180):
				print(grad_angle[i][j])
	d = np.array(mag_image, dtype=np.uint8)
	
	hog_vector = []
	hog1 = []
	hog2 = []
	hog3 = []
	hog4 = []
	hog = [0,0,0,0,0,0,0,0,0]

	mag_image = np.pad(mag_image, ((1,1),(1,1)), 'constant')
	grad_angle = np.pad(grad_angle, ((1,1),(1,1)), 'constant')
	


	difference = (new_height)%8
	new_h=(new_height)-difference

	difference = (new_width)%8
	new_w=(new_width)-difference

    #Each Block Computation
	for i in range(0,new_h-15,8):
		for j in range(0,new_w-15,8):

		    # Top Left Cell Computation of Cuurent Block
			for k in range(i,i+8):
				for l in range(j,j+8):
					#Bin computations within each cell
					if(grad_angle[k][l] <= 0):
						right_distance = 0 - grad_angle[k][l]
						left_distance = 10 + (10 + grad_angle[k][l])
						hog[0] = hog[0]+((left_distance/20) * mag_image[k][l])
						hog[8] = hog[8]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 160):
						right_distance = 10 + (170 - grad_angle[k][l])
						left_distance = grad_angle[k][l] - 160
						hog[0] = hog[0]+((left_distance/20) * mag_image[k][l])
						hog[8] = hog[8]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 0 and grad_angle[k][l] <= 20):
						left_distance = grad_angle[k][l] - 0
						right_distance = 20 - grad_angle[k][l]
						hog[1] = hog[1]+((left_distance/20) * mag_image[k][l])
						hog[0] = hog[0]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 20 and grad_angle[k][l] <= 40):
						left_distance = grad_angle[k][l] - 20
						right_distance = 40 - grad_angle[k][l]
						hog[2] = hog[2]+((left_distance/20) * mag_image[k][l])
						hog[1] = hog[1]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 40 and grad_angle[k][l] <= 60):
						left_distance = grad_angle[k][l] - 40
						right_distance = 60 - grad_angle[k][l]
						hog[3] = hog[3]+((left_distance/20) * mag_image[k][l])
						hog[2] = hog[2]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 60 and grad_angle[k][l] <= 80):
						left_distance = grad_angle[k][l] - 60
						right_distance = 80 - grad_angle[k][l]
						hog[4] = hog[4]+((left_distance/20) * mag_image[k][l])
						hog[3] = hog[3]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 80 and grad_angle[k][l] <= 100):
						left_distance = grad_angle[k][l] - 80
						right_distance = 100 - grad_angle[k][l]
						hog[5] = hog[5]+((left_distance/20) * mag_image[k][l])
						hog[4] = hog[4]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 100 and grad_angle[k][l] <= 120):
						left_distance = grad_angle[k][l] - 100
						right_distance = 120 - grad_angle[k][l]
						hog[6] = hog[6]+((left_distance/20) * mag_image[k][l])
						hog[5] = hog[5]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 120 and grad_angle[k][l] <= 140):
						left_distance = grad_angle[k][l] - 120
						right_distance = 140 - grad_angle[k][l]
						hog[7] = hog[7]+((left_distance/20) * mag_image[k][l])
						hog[6] = hog[6]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 140 and grad_angle[k][l] <= 160):
						left_distance = grad_angle[k][l] - 140
						right_distance = 160 - grad_angle[k][l]
						hog[8] = hog[8]+((left_distance/20) * mag_image[k][l])
						hog[7] = hog[7]+((right_distance/20) * mag_image[k][l])
			hog1 = hog
			hog = [0,0,0,0,0,0,0,0,0]

		#	print("End of cell.............")

            # Top right Cell Computation of Cuurent Block
			for k in range(i,i+8):
				for l in range(j+8,j+16):

					if(grad_angle[k][l] <= 0):
						right_distance = 0 - grad_angle[k][l]
						left_distance = 10 + (10 + grad_angle[k][l])
						hog[0] = hog[0]+((left_distance/20) * mag_image[k][l])
						hog[8] = hog[8]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 160):
						right_distance = 10 + (170 - grad_angle[k][l])
						left_distance = grad_angle[k][l] - 160
						hog[0] = hog[0]+((left_distance/20) * mag_image[k][l])
						hog[8] = hog[8]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 0 and grad_angle[k][l] <= 20):
						left_distance = grad_angle[k][l] - 0
						right_distance = 20 - grad_angle[k][l]
						hog[1] = hog[1]+((left_distance/20) * mag_image[k][l])
						hog[0] = hog[0]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 20 and grad_angle[k][l] <= 40):
						left_distance = grad_angle[k][l] - 20
						right_distance = 40 - grad_angle[k][l]
						hog[2] = hog[2]+((left_distance/20) * mag_image[k][l])
						hog[1] = hog[1]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 40 and grad_angle[k][l] <= 60):
						left_distance = grad_angle[k][l] - 40
						right_distance = 60 - grad_angle[k][l]
						hog[3] = hog[3]+((left_distance/20) * mag_image[k][l])
						hog[2] = hog[2]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 60 and grad_angle[k][l] <= 80):
						left_distance = grad_angle[k][l] - 60
						right_distance = 80 - grad_angle[k][l]
						hog[4] = hog[4]+((left_distance/20) * mag_image[k][l])
						hog[3] = hog[3]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 80 and grad_angle[k][l] <= 100):
						left_distance = grad_angle[k][l] - 80
						right_distance = 100 - grad_angle[k][l]
						hog[5] = hog[5]+((left_distance/20) * mag_image[k][l])
						hog[4] = hog[4]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 100 and grad_angle[k][l] <= 120):
						left_distance = grad_angle[k][l] - 100
						right_distance = 120 - grad_angle[k][l]
						hog[6] = hog[6]+((left_distance/20) * mag_image[k][l])
						hog[5] = hog[5]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 120 and grad_angle[k][l] <= 140):
						left_distance = grad_angle[k][l] - 120
						right_distance = 140 - grad_angle[k][l]
						hog[7] = hog[7]+((left_distance/20) * mag_image[k][l])
						hog[6] = hog[6]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 140 and grad_angle[k][l] <= 160):
						left_distance = grad_angle[k][l] - 140
						right_distance = 160 - grad_angle[k][l]
						hog[8] = hog[8]+((left_distance/20) * mag_image[k][l])
						hog[7] = hog[7]+((right_distance/20) * mag_image[k][l])
			hog2 = hog
			hog = [0,0,0,0,0,0,0,0,0]
			
		#	print("End of cell.............")
					
             # Bottom Left Cell Computation of Cuurent Block
			for k in range(i+8,i+16):
				for l in range(j,j+8):
					
					if(grad_angle[k][l] <= 0):
						right_distance = 0 - grad_angle[k][l]
						left_distance = 10 + (10 + grad_angle[k][l])
						hog[0] = hog[0]+((left_distance/20) * mag_image[k][l])
						hog[8] = hog[8]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 160):
						right_distance = 10 + (170 - grad_angle[k][l])
						left_distance = grad_angle[k][l] - 160
						hog[0] = hog[0]+((left_distance/20) * mag_image[k][l])
						hog[8] = hog[8]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 0 and grad_angle[k][l] <= 20):
						left_distance = grad_angle[k][l] - 0
						right_distance = 20 - grad_angle[k][l]
						hog[1] = hog[1]+((left_distance/20) * mag_image[k][l])
						hog[0] = hog[0]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 20 and grad_angle[k][l] <= 40):
						left_distance = grad_angle[k][l] - 20
						right_distance = 40 - grad_angle[k][l]
						hog[2] = hog[2]+((left_distance/20) * mag_image[k][l])
						hog[1] = hog[1]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 40 and grad_angle[k][l] <= 60):
						left_distance = grad_angle[k][l] - 40
						right_distance = 60 - grad_angle[k][l]
						hog[3] = hog[3]+((left_distance/20) * mag_image[k][l])
						hog[2] = hog[2]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 60 and grad_angle[k][l] <= 80):
						left_distance = grad_angle[k][l] - 60
						right_distance = 80 - grad_angle[k][l]
						hog[4] = hog[4]+((left_distance/20) * mag_image[k][l])
						hog[3] = hog[3]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 80 and grad_angle[k][l] <= 100):
						left_distance = grad_angle[k][l] - 80
						right_distance = 100 - grad_angle[k][l]
						hog[5] = hog[5]+((left_distance/20) * mag_image[k][l])
						hog[4] = hog[4]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 100 and grad_angle[k][l] <= 120):
						left_distance = grad_angle[k][l] - 100
						right_distance = 120 - grad_angle[k][l]
						hog[6] = hog[6]+((left_distance/20) * mag_image[k][l])
						hog[5] = hog[5]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 120 and grad_angle[k][l] <= 140):
						left_distance = grad_angle[k][l] - 120
						right_distance = 140 - grad_angle[k][l]
						hog[7] = hog[7]+((left_distance/20) * mag_image[k][l])
						hog[6] = hog[6]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 140 and grad_angle[k][l] <= 160):
						left_distance = grad_angle[k][l] - 140
						right_distance = 160 - grad_angle[k][l]
						hog[8] = hog[8]+((left_distance/20) * mag_image[k][l])
						hog[7] = hog[7]+((right_distance/20) * mag_image[k][l])
			hog3 = hog
			hog = [0,0,0,0,0,0,0,0,0]
					
            # Bottom right Cell Computation of Cuurent Block
			for k in range(i+8,i+16):
				for l in range(j+8,j+16):
					if(grad_angle[k][l] <= 0):
						right_distance = 0 - grad_angle[k][l]
						left_distance = 10 + (10 + grad_angle[k][l])
						hog[0] = hog[0]+((left_distance/20) * mag_image[k][l])
						hog[8] = hog[8]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 160):
						right_distance = 10 + (170 - grad_angle[k][l])
						left_distance = grad_angle[k][l] - 160
						hog[0] = hog[0]+((left_distance/20) * mag_image[k][l])
						hog[8] = hog[8]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 0 and grad_angle[k][l] <= 20):
						left_distance = grad_angle[k][l] - 0
						right_distance = 20 - grad_angle[k][l]
						hog[1] = hog[1]+((left_distance/20) * mag_image[k][l])
						hog[0] = hog[0]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 20 and grad_angle[k][l] <= 40):
						left_distance = grad_angle[k][l] - 20
						right_distance = 40 - grad_angle[k][l]
						hog[2] = hog[2]+((left_distance/20) * mag_image[k][l])
						hog[1] = hog[1]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 40 and grad_angle[k][l] <= 60):
						left_distance = grad_angle[k][l] - 40
						right_distance = 60 - grad_angle[k][l]
						hog[3] = hog[3]+((left_distance/20) * mag_image[k][l])
						hog[2] = hog[2]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 60 and grad_angle[k][l] <= 80):
						left_distance = grad_angle[k][l] - 60
						right_distance = 80 - grad_angle[k][l]
						hog[4] = hog[4]+((left_distance/20) * mag_image[k][l])
						hog[3] = hog[3]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 80 and grad_angle[k][l] <= 100):
						left_distance = grad_angle[k][l] - 80
						right_distance = 100 - grad_angle[k][l]
						hog[5] = hog[5]+((left_distance/20) * mag_image[k][l])
						hog[4] = hog[4]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 100 and grad_angle[k][l] <= 120):
						left_distance = grad_angle[k][l] - 100
						right_distance = 120 - grad_angle[k][l]
						hog[6] = hog[6]+((left_distance/20) * mag_image[k][l])
						hog[5] = hog[5]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 120 and grad_angle[k][l] <= 140):
						left_distance = grad_angle[k][l] - 120
						right_distance = 140 - grad_angle[k][l]
						hog[7] = hog[7]+((left_distance/20) * mag_image[k][l])
						hog[6] = hog[6]+((right_distance/20) * mag_image[k][l])
					elif(grad_angle[k][l] > 140 and grad_angle[k][l] <= 160):
						left_distance = grad_angle[k][l] - 140
						right_distance = 160 - grad_angle[k][l]
						hog[8] = hog[8]+((left_distance/20) * mag_image[k][l])
						hog[7] = hog[7]+((right_distance/20) * mag_image[k][l])
			hog4 = hog

			hog = [0,0,0,0,0,0,0,0,0]
			normalize = hog1 + hog2 + hog3 + hog4
			nsum = 0
			for pp in range(0,len(normalize)):
				nsum = nsum + (normalize[pp]*normalize[pp])
			
			#L2 Block Normalization
			normalize_value = np.sqrt(nsum)
			for mn in range(0,9):
				if normalize_value !=0:
					hog1[mn] = hog1[mn]/normalize_value
					hog2[mn] = hog2[mn]/normalize_value
					hog3[mn] = hog3[mn]/normalize_value
					hog4[mn] = hog4[mn]/normalize_value
				else:
					continue
			final = hog1 + hog2 + hog3 + hog4
			hog_vector = hog_vector + hog1 + hog2 + hog3 + hog4
	#np.savetxt('ktshouldwork.csv', hog_vector, fmt="%Lf", delimiter=",")
    #Return the final hog decriptor for the input image to Neural_hog.py
	return hog_vector