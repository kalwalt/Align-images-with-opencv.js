// This code is based on the article here: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
// You can see the c++ commented from the article above the Javascript
// original code (modified by @kalwalt) from blog post https://scottsuhy.com/2021/02/01/image-alignment-feature-based-in-opencv-js-javascript/

function Align_img() {

    let detector_option = document.getElementById('detector').value;
    let match_option = document.getElementById('match').value;
    let matchDistance_option = document.getElementById('distance').value;
    let knnDistance_option = document.getElementById('knn_distance').value;
    //let knnDistance_option = 0.75

    image_A_element = document.getElementById('form');
    image_B_element = document.getElementById('scanned-form');

    //im2 is the original reference image we are trying to align to
    let im2 = cv.imread(image_A_element);
    //console.log(im2);

    //im1 is the image we are trying to line up correctly
    let im1 = cv.imread(image_B_element);
    //console.log(im1);

    //17            Convert images to grayscale
    //18            Mat im1Gray, im2Gray;
    //19            cvtColor(im1, im1Gray, CV_BGR2GRAY);
    //20            cvtColor(im2, im2Gray, CV_BGR2GRAY);
    let im1Gray = new cv.Mat();
    let im2Gray = new cv.Mat();
    cv.cvtColor(im1, im1Gray, cv.COLOR_BGRA2GRAY);
    cv.cvtColor(im2, im2Gray, cv.COLOR_BGRA2GRAY);
    //console.log(im1Gray);


    //22            Variables to store keypoints and descriptors
    //23            std::vector<KeyPoint> keypoints1, keypoints2;
    //24            Mat descriptors1, descriptors2;
    let keypoints1 = new cv.KeyPointVector();
    let keypoints2 = new cv.KeyPointVector();
    let descriptors1 = new cv.Mat();
    let descriptors2 = new cv.Mat();

    //26            Detect ORB features and compute descriptors.
    //27            Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
    //28            orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
    //29            orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

    if (detector_option == 0) {
        var orb = new cv.ORB(5000);
    } else if (detector_option == 1) {
        var orb = new cv.AKAZE();
    }

    orb.detectAndCompute(im1Gray, new cv.Mat(), keypoints1, descriptors1);
    orb.detectAndCompute(im2Gray, new cv.Mat(), keypoints2, descriptors2);

    console.log("Total of ", keypoints1.size(), " keypoints1 (img to align) and ", keypoints2.size(), " keypoints2 (reference)");
    console.log("here are the first 5 keypoints for keypoints1:");
    for (let i = 0; i < keypoints1.size(); i++) {
        console.log("keypoints1: [", i, "]", keypoints1.get(i).pt.x, keypoints1.get(i).pt.y);
        if (i === 5) { break; }
    }

    //31            Match features.
    //32            std::vector<DMatch> matches;
    //33            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //34            matcher->match(descriptors1, descriptors2, matches, Mat());

    let good_matches = new cv.DMatchVector();

    if (match_option == 0) {//match
        var bf = new cv.BFMatcher(cv.NORM_HAMMING, true);

        var matches = new cv.DMatchVector();
        bf.match(descriptors1, descriptors2, matches);

        //36            Sort matches by score
        //37            std::sort(matches.begin(), matches.end());
        //39            Remove not so good matches
        //40            const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
        //41            matches.erase(matches.begin()+numGoodMatches, matches.end());
        console.log("good_matches: ", good_matches);
        console.log("matches.size: ", matches.size());
        for (let i = 0; i < matches.size(); i++) {
            if (matches.get(i).distance < matchDistance_option) {
                good_matches.push_back(matches.get(i));
            }
        }
        if (good_matches.size() <= 3) {
            alert("Less than 4 matches found!");
            return;
        }
    }
    else if (match_option == 1) { //knnMatch
        var bf = new cv.BFMatcher();
        var matches = new cv.DMatchVectorVector();

        bf.knnMatch(descriptors1, descriptors2, matches, 2);

        let counter = 0;
        for (let i = 0; i < matches.size(); ++i) {
            let match = matches.get(i);
            let dMatch1 = match.get(0);
            let dMatch2 = match.get(1);
           // console.log(knnDistance_option);

            //console.log("[", i, "] ", "dMatch1: ", dMatch1, "dMatch2: ", dMatch2);
            if (dMatch1.distance <= dMatch2.distance * parseFloat(knnDistance_option)) {
                //console.log("***Good Match***", "dMatch1.distance: ", dMatch1.distance, "was less than or = to: ", "dMatch2.distance * parseFloat(knnDistance_option)", dMatch2.distance * parseFloat(knnDistance_option), "dMatch2.distance: ", dMatch2.distance, "knnDistance", knnDistance_option);
                good_matches.push_back(dMatch1);
                counter++;
            }
        }

        console.log("keeping ", counter, " points in good_matches vector out of ", matches.size(), " contained in this match vector:", matches);
        console.log("here are first 5 matches");
        for (let t = 0; t < matches.size(); ++t) {
            console.log("[" + t + "]", "matches: ", matches.get(t));
            if (t === 5) { break; }
        }

        console.log("here are first 5 good_matches");
        for (let r = 0; r < good_matches.size(); ++r) {
            console.log("[" + r + "]", "good_matches: ", good_matches.get(r));
            if (r === 5) { break; }
        }
    }

    //44            Draw top matches
    //45            Mat imMatches;
    //46            drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
    //47            imwrite("matches.jpg", imMatches);
    let imMatches = new cv.Mat();
    let color = new cv.Scalar(0, 255, 0, 255);
    cv.drawMatches(im1, keypoints1, im2, keypoints2, good_matches, imMatches, color);
    cv.imshow('imageCompareMatches', imMatches);

    let keypoints1_img = new cv.Mat();
    let keypoints2_img = new cv.Mat();
    let keypointcolor = new cv.Scalar(0, 255, 0, 255);
    cv.drawKeypoints(im1Gray, keypoints1, keypoints1_img, keypointcolor);
    cv.drawKeypoints(im2Gray, keypoints2, keypoints2_img, keypointcolor);

    //cv.imshow('keypoints1', keypoints1_img);
    //cv.imshow('keypoints2', keypoints2_img);

    //50            Extract location of good matches
    //51            std::vector<Point2f> points1, points2;
    //53            for( size_t i = 0; i < matches.size(); i++ )
    //54            {
    //55                points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
    //56                points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
    //57            }
    let points1 = [];
    let points2 = [];
    /*for (let i = 0; i < good_matches.size(); i++) {
        points1.push(keypoints1.get(good_matches.get(i).queryIdx).pt);
        points2.push(keypoints2.get(good_matches.get(i).trainIdx).pt);
    }*/

    for (let i = 0; i < good_matches.size(); i++) {
        points1.push(keypoints1.get(good_matches.get(i).queryIdx).pt.x);
        points1.push(keypoints1.get(good_matches.get(i).queryIdx).pt.y);
        points2.push(keypoints2.get(good_matches.get(i).trainIdx).pt.x);
        points2.push(keypoints2.get(good_matches.get(i).trainIdx).pt.y);
    }

    console.log("points1:", points1, "points2:", points2);

    //59            Find homography
    //60            h = findHomography( points1, points2, RANSAC );
    //let mat1 = cv.matFromArray(points1.length, 2, cv.CV_32F, points1);
    //let mat2 = cv.matFromArray(points2.length, 2, cv.CV_32F, points2); //32FC2
    console.log(points1.length);
    
    var mat1 = new cv.Mat(points1.length/2,1,cv.CV_32FC2);
    mat1.data32F.set(points1);
    var mat2 = new cv.Mat(points2.length/2,1,cv.CV_32FC2);
    mat2.data32F.set(points2);
    console.log("mat1: ", mat1, "mat2: ", mat2);
    //Reference: https://docs.opencv.org/3.3.0/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    let h = cv.findHomography(mat1, mat2, cv.RANSAC);
    /*if (h.empty()) {
        alert("homography matrix empty!");
        return;
    }
    else { console.log("h:", h); }*/
    if (h.empty())
        {
            alert("homography matrix empty!");
            return;
        }
        else{
            console.log("h:", h);
            console.log("[", h.data64F[0],",", h.data64F[1], ",", h.data64F[2]);
            console.log("", h.data64F[3],",", h.data64F[4], ",", h.data64F[5]);
            console.log("", h.data64F[6],",", h.data64F[7], ",", h.data64F[8], "]");
        }

    //62          Use homography to warp image
    //63          warpPerspective(im1, im1Reg, h, im2.size());
    //Reference: https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    let image_B_final_result = new cv.Mat();
    cv.warpPerspective(im1, image_B_final_result, h, im2.size());
    cv.imshow('imageAligned', image_B_final_result);


    matches.delete();
    bf.delete();
    orb.delete();
    descriptors1.delete();
    descriptors2.delete();
    keypoints1.delete();
    keypoints2.delete();
    im1Gray.delete();
    im2Gray.delete();
    h.delete();
    image_B_final_result.delete();
    mat1.delete();
    mat2.delete();
}