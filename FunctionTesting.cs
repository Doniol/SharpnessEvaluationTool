using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sharpness
{
    public static class FunctionTesting
    {
        public static void TestFunction(TestSet set)
        /*  This function tests all created functions using the given testset */
        {
            // Create array for all the to be executed algorithms
            TestAlgorithm[] allAlgo = new TestAlgorithm[13] {
                new TestAlgorithm(Functions.Laplace, "laplace"),
                new TestAlgorithm(Functions.BlurLaplace, "blurLaplace"),
                new TestAlgorithm(Functions.BlurLaplaceSum, "blurLaplaceSum"),
                new TestAlgorithm(Functions.BlurLaplaceVar, "blurLaplaceVar"),
                new TestAlgorithm(Functions.SpecialSobel, "specialSobel"),
                new TestAlgorithm(Functions.BaseSobel, "baseSobel"),
                new TestAlgorithm(Functions.TenegradSobel, "tenegradSobel"),
                new TestAlgorithm(Functions.SobelVariance, "sobelVariance"),
                new TestAlgorithm(Functions.Canny, "canny"),
                new TestAlgorithm(Functions.GreyVariance, "greyVariance"),
                new TestAlgorithm(Functions.EdgeWidth, "edgeWidth"),
                new TestAlgorithm(Functions.ExistingSharpness, "existingSharpness"),
                new TestAlgorithm(Functions.ExistingOldSharpness, "existingOldSharpness") };

            // Loop through all images in given set
            foreach (string uncropped in Directory.GetFiles(set.Uncrop))
            {
                // Crop selected image
                string imageDst = set.Crop + Path.DirectorySeparatorChar +
                    uncropped.Split(Path.DirectorySeparatorChar).Last();
                imageDst = imageDst.Replace("Uncropped", "Cropped");

                // Check whether an image has been cropped, if not, it's because
                //  the particle is too vague to detects, and the image is ignored
                bool hasCropped = CropImage(uncropped, imageDst, set.Background);
                // Keep track of amount of images without particle within set
                if (!hasCropped)
                {
                    set.MissingParticle++;
                }
                else
                {
                    // Loop through all selected functions
                    foreach (TestAlgorithm algo in allAlgo)
                    {
                        // Console.WriteLine(algo.Name);
                        algo.ExecuteAlgorithm(imageDst);
                    }
                }
            }
            Console.WriteLine($"Images without particles: {set.MissingParticle}");
            // Store algorithms and their results within given set
            set.Algorithms = allAlgo;
        }

        public static void RankResults(TestSet set)
        /*  Function for creating new folders with images for each tested
            algorithm. Theses images are the same ones that have been tested,
            but have now been rearranged and renamed to reflect their ranking
            by the algorithm. */
        {
            // Loop through all tested algorithms
            foreach (TestAlgorithm algo in set.Algorithms)
            {
                // Create new folder for selected algorithm
                string newDir = set.Results + Path.DirectorySeparatorChar +
                    $"{algo.Name}";
                if (Directory.Exists(newDir))
                {
                    Directory.Delete(newDir, true);
                }
                Directory.CreateDirectory(newDir);
                // Sort algorithm results by value
                Dictionary<string, double> sortedDict = algo.Results.
                    OrderBy(x => x.Value).ToDictionary(x => x.Key, x => x.Value);

                // Loop through the sorted images within the dictionary
                int i = 0;
                foreach (KeyValuePair<string, double> kvp in sortedDict)
                {
                    // Rename and store the selected image
                    string imageLoc = newDir + Path.DirectorySeparatorChar +
                        $"{i}-{kvp.Value}.jpg";
                    File.Copy(kvp.Key, imageLoc);
                    i++;
                }
            }
        }

        public static bool CropImage(string image, string dst, string background,
            int threshold = 4)
        /*  This function crops given images to a (hopefully) smaller size.
            By vaguely detecting where within the image the particle is located,
            a smaller, quicker to analyse, version of the original image is 
            cropped out around the particle. Returns a bool defining whether or
            not a cropped image has been created. */
        {
            // Based on old branch-code in CytoCV.cs
            // Grayscale the images
            Mat backgroundImage = Cv2.ImRead(background);
            Cv2.CvtColor(backgroundImage, backgroundImage,
                ColorConversionCodes.RGB2GRAY);
            Mat src = Cv2.ImRead(image);
            Cv2.CvtColor(src, src, ColorConversionCodes.RGB2GRAY);

            // Create buffer for storing temporary results
            Mat buffer = new Mat(backgroundImage.Size(), backgroundImage.Type());
            // Create kernel for morphological dilation
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse,
                new OpenCvSharp.Size(5, 5));

            // Calculate and store absolute difference between fore- and background
            Cv2.Absdiff(src, backgroundImage, buffer);
            // Find global minimum and maximum absolute differences
            Cv2.MinMaxIdx(buffer, out double min, out double max);

            // Morphologically erode, and then dilate, image; removes noise
            Cv2.MorphologyEx(buffer, buffer, MorphTypes.Open, kernel);
            // Apply threshold on Absdiff-image
            // Threshold is normally 4, but can vary if 4 is too high to detect
            //  a particle with
            Cv2.Threshold(buffer, buffer, min + (max - min) / threshold, 255,
                ThresholdTypes.Binary);

            // Finds contours in thresholded image
            Point[][] contours = Cv2.FindContoursAsArray(buffer, RetrievalModes.External,
                ContourApproximationModes.ApproxSimple);

            // Calculate the minimal boundingbox in which the image-particle fits
            Rect boundingBox = new Rect();
            // Loop through all detected contours, and update box to fit all of them
            foreach (Point[] contour in contours)
            {
                if (boundingBox.Width == 0)
                    boundingBox = Cv2.BoundingRect(contour);
                else
                    boundingBox |= Cv2.BoundingRect(contour);
            }
            // If there is no boundingbox, there are no contours
            // If the boundingbox is too large, incorrect particles have
            //  been detected; bead particles are relatively small
            if (boundingBox.Width == 0 || boundingBox.Width > 120 ||
                boundingBox.Height > 120)
            {
                // Option 1: retry with lower threshold to have an easier time
                //  finding particles; Sometimes results in larger images that 
                //  take a helluva lot longer to process
                //CropImage(image, dst, background, threshold + 1);
                // Option 2: do not crop image
                //  Same issue as option 1, but worse
                //src.SaveImage(dst);
                // Option 3: ignore image, and do not process it using the
                //  test-algorithms; No long processing times, but also less
                //  images on which to base observations
                return false;
            }
            // Increase boundingbox by 40px on each side
            // This is done incase there are tiny bits of the particle that
            //  may otherwise fall outside of the boundingBox
            boundingBox.Inflate(40, 40);

            // Create image boundary the size of the background image
            Rect imageBounds = new Rect(new Point(0, 0), src.Size());
            // Crop boundingbox to imageBounds limits
            // I assume this is incase the +40 inflation causes the image to 
            //  become larger than the original; under any normal circumstances,
            //  this should never be the case. If something weird happens,
            //  however, this failsafe might be useful in avoiding out-of-bounds
            //  errors.
            boundingBox &= imageBounds;

            // Crop image according to boundingbox
            // Copped image contains cropped version of original image
            Mat croppedImage = new Mat(src, boundingBox);
            croppedImage.SaveImage(dst);
            return true;
        }
    }
}
