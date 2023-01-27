using System;
using System.IO;
using TestAlgorithm = AlgorithmClass.TestAlgorithm;

namespace SetClass
{
    public class TestSet
    /*  Class for handling testsets. Contains all paths to relevant
        directories, and the algorithms that have tested it. */
    {
        public string Set { get; set; }
        public string Crop { get; set; }
        public string Uncrop { get; set; }
        public string Background { get; set; }
        public string Name { get; set; }
        public string Results { get; set; }
        public global::AlgorithmClass.TestAlgorithm[] Algorithms { get; set; }

        public TestSet(string directory, string name)
        /*  Select base directory for the testset. Within this directory, all
            images and results can be found. The directory that is selected,
            should already contain all necessary folders, and must contain both 
            the desired uncropped images within the "uncropped"-subfolder, and
            the corresponding background-image. 
            The given name is arbitrary, and just useful for recognising the 
            testset later on. */
        {
            this.Set = directory;
            this.Crop = directory + Path.DirectorySeparatorChar + "cropped";
            this.Uncrop = directory + Path.DirectorySeparatorChar + "uncropped";
            this.Background = directory + Path.DirectorySeparatorChar + "background.jpg";
            this.Name = name;
            this.Results = directory + Path.DirectorySeparatorChar + "results";
        }
    }
}
