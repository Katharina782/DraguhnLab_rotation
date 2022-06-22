# DraguhnLab_rotation
Analysis scripts of whole-cell patch-clamp recordings.


**Cellchar**
* First AP fired within 10ms was analysed
* Interspike interval was calculated for the first current injection with at least 7 action potentials. Difference between 6th and 7th peak was divided by difference between 1st and 2nd peak
* Ih sag was calculated by dividing the max. hyperpolarization by the stable hyperpolarization that follows 
*	Input-output curve was drawn by plotting the number of action potentials vs the currentinjection

**Sine wave/phase precision** 
* Using different frequencies, investigate how precise action potentials are fired in phase

**Spulse protocol**
* Injecting a short pulse investigate the cell's response

**Circular statistics:**
*	The counts in the polar histograms represents the square root of the number of counts. This way the area represents the number of counts.
*	To calculate the mean of means (taking into accounts the vector length of each individual mean vector) the function mean_of_means() was used
*	To calculate the angle from rectangular coordinates, different inverse functions (arctan, arccos, arsin) have to be used depending on the quadrant of the circle the angle is expected to be in. 


**Data manipulation**

Here, I combine the Spulse, Cellchar & sine wave recordings, as well as the handwritten parameter excel sheet and the histological measurements into one dataframe for further statistical analysis.

**Stats**

Collection of all statistical analysis performed on the data
