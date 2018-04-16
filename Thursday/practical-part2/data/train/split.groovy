dirs = new File(".").listFiles()

for (dir in dirs)
	if (dir.isDirectory()) {
		odir = new File("../valid/"+dir.toString())
		odir.mkdirs()

		files = dir.listFiles() as List
		Collections.shuffle(files)
		for (i=0; i<0.1*files.size(); i++) {
			files[i].renameTo(new File(odir, files[i].getName()))
		}
	}