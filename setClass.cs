using System;
using System.IO;
using testAlgorithm = algorithmClass.testAlgorithm;

public class setClass
{
    public class testSet
    /*  Class for handling testsets. Contains all paths to relevant
        directories, and the algorithms that have tested it. */
    {
        public string set;
        public string crop;
        public string uncrop;
        public string background;
        public string name;
        public string results;
        public testAlgorithm[] algorithms;

        public testSet(string directory, string name)
        /*  Select base directory for the testset. Within this directory, all
            images and results can be found. The directory that is selected,
            should already contain all necessary folders, and must contain both 
            the desired uncropped images within the "uncropped"-subfolder, and
            the corresponding background-image. 
            The given name is arbitrary, and just useful for recognising the 
            testset later on. */
        {
            this.set = directory;
            this.crop = directory + Path.DirectorySeparatorChar + "cropped";
            this.uncrop = directory + Path.DirectorySeparatorChar + "uncropped";
            this.background = directory + Path.DirectorySeparatorChar + "background.jpg";
            this.name = name;
            this.results = directory + Path.DirectorySeparatorChar + "results";
        }
    }
}
