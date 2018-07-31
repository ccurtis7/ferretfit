prefix = 'W_P5_R3'
folder = 'C:\\Users\\enuser\\Desktop\\ferret_tracking\\wonky\\'

function track_extract(folder, prefix) {
	
	run("Close All");
	fullfilename = folder + prefix + '.png';
	filename = prefix + '.png';
	open(fullfilename);

	run("Split Channels");
	blue = filename + ' (blue)';
	green = filename + ' (green)';
	red = filename + ' (red)';

	selectWindow(blue);
	run('Close');
	imageCalculator("OR create", red, green); 
	bandw = 'Result of ' + red;

	selectWindow(bandw);
	run("Threshold...");
	setThreshold(60, 255);
	setOption('BlackBackground', false);
	run("Convert to Mask");
	run("Close");

	padded = 'Padded ' + bandw
	run("Image Padder", "pad_top=10 pad_left=10 pad_right=10 pad_bottom=10");
	selectWindow('Padded Result')
	run("Invert");
	run("Fill Holes");
	run("Erode");
	run("Erode");
	run("Erode");

	run("Dilate");
	run("Dilate");
	run("Dilate");

	run("Shape Smoothing", "relative_proportion_fds=2 absolute_number_fds=9 keep=[Absolute_number of FDs]");
	run("Watershed");
	clean = folder + prefix + '_c.png';
	saveAs("PNG", clean);
	run("Analyze Particles...", "display clear summarize add");
	results = folder + prefix + '.csv';
	selectWindow('Results');
	saveAs("Results", results);
	run("Close All");
	selectWindow('Results');
	run('Close');
	selectWindow('Summary');
	run('Close');
	selectWindow('ROI Manager');
	run('Close');

	open(fullfilename);
	open(clean);

	
}

track_extract(folder, prefix)

