package main

import (
	"bufio"
	"fmt"
	_ "image/png"
	"io/ioutil"
	"log"
	"os"
	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

const (
	labelsFile = "pupil_labels.txt"
	// Capture frame width
	W = 300
	// Capture frame height
	H = 300
)

var (
	// global TF session, re-usable and concurrency safe
	//session *tf.Session
	// global model graph
	//graph *tf.Graph
	// global slice of labels
	labels []string
)

func main() {
	/*
		modelDir := flag.String("dir", "", "Directory containing the trained model files.")
		device := flag.Int("device", 0, "The webcam device ID")
		flag.Parse()
		if *modelDir == "" {
			flag.Usage()
			return
		}
		//deviceID = *device
	*/
	//modelFile := filepath.Join(*modelDir, defaultModelFilename)

	model, err := ioutil.ReadFile("frozen_pupil.pb")
	if err != nil {
		log.Fatalf("Cannot load frozen model: %v", err)
	}

	loadLabels(labelsFile)

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()
	//img := gocv.IMRead("frameGR_1.png", gocv.IMReadColor)
	fmt.Println("zoinks=============================================")
	//pixelgl.Run(run)
	tensor, err := makeTensorFromImage("frameGR_57.jpeg")
	if err != nil {
		log.Fatalf("error making input tensor: %v", err)
	}
	fmt.Println("-----------------------------------------------------------")

	start := time.Now()
	fmt.Println("okayhere6----------------------------------")
	//inputop := graph.Operation("image_tensor")
	fmt.Println("okayhere7----------------------------------")
	// Output ops
	//o1 := graph.Operation("detection_boxes")
	//o2 := graph.Operation("detection_scores")
	//o3 := graph.Operation("detection_classes")
	//o4 := graph.Operation("num_detections")
	fmt.Println("okayhere8----------------------------------")

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("image_tensor").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("detection_boxes").Output(0)}, nil)
	//graph.Operation("detection_scores").Output(0),
	//graph.Operation("detection_classes").Output(0),
	//graph.Operation("num_detections").Output(0),
	//},
	//nil)
	if err != nil {
		log.Fatalf("Error running session: %v", err)
	}
	elapsed := time.Since(start)
	log.Printf("Inference took %s", elapsed)
	fmt.Println("okayhere9----------------------------------")
	//probabilities := output[1].Value().([][]float32)[0]
	//classes := output[2].Value().([][]float32)[0]
	boxes := output[0].Value().([][][]float32)[0]
	//fmt.Println(tensor)

	// Run inference on the newly made input tensor
	//probabilities, classes, boxes := predictObjectBoxes(tensor)
	/*
		if err != nil {
			log.Fatalf("error making prediction: %v", err)
		}
	*/
	//fmt.Println(numdetect)
	//fmt.Println(probabilities[0])
	fmt.Println(boxes[0])
	//fmt.Println(classes[0])
	//fmt.Println(classes)
	//fmt.Println(reflect.TypeOf(boxes))
	//fmt.Println(len(boxes))
}

func loadLabels(labelsFile string) {
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
}

// Read frames from the webcam in a goroutine to absorb some of the read latency.
// This helps gain ~2 FPS.
/*
func capture(deviceID int, frames chan []byte) {
	cam, err := gocv.VideoCaptureDevice(deviceID)
	if err != nil {
		log.Fatal("failed reading cam")
	}
	defer cam.Close()

	frame := gocv.NewMat()
	defer frame.Close()

	for {
		if ok := cam.Read(frame); !ok {
			log.Fatal("failed reading cam")
		}

		// Resize the Mat in place.
		gocv.Resize(frame, frame, image.Point{X: W, Y: H}, 0.0, 0.0, gocv.InterpolationNearestNeighbor)

		// Encode Mat as a bmp (uncompressed)
		buf, err := gocv.IMEncode(".bmp", frame)
		if err != nil {
			log.Fatalf("Error encoding frame: %v", err)
		}

		// Push the frame to the channel
		frames <- buf
	}
}
*/

func run() {
	// Setup Pixel window
	/*
			cfg := pixelgl.WindowConfig{
				Title:  "Thinger",
				Bounds: pixel.R(0, 0, W, H),
				VSync:  true,
			}
			win, err := pixelgl.NewWindow(cfg)
			if err != nil {
				panic(err)
			}

			// Start up the background capture
			framesChan := make(chan []byte)
			go capture(deviceID, framesChan)

			// Setup Pixel requirements for drawing boxes and labels
			mat := pixel.IM
			mat = mat.Moved(win.Bounds().Center())

			atlas := text.NewAtlas(basicfont.Face7x13, text.ASCII)
			imd := imdraw.New(nil)

			// Some local vars to calculate frame rate
			var (
				frames = 0
				second = time.Tick(time.Second)
			)

			for !win.Closed() {
				select {
				// Run inference if we have a new frame to read
				case frame := <-framesChan:

		// Make a tensor and an Image from the frame bytes

		tensor, _, err := makeTensorFromImage("frameGR_1.png")
		if err != nil {
			log.Fatalf("error making input tensor: %v", err)
		}

		// Run inference on the newly made input tensor
		probabilities, classes, boxes, err := predictObjectBoxes(tensor)
		if err != nil {
			log.Fatalf("error making prediction: %v", err)
		}

		fmt.Println(probabilities)
		fmt.Println(classes)
		fmt.Println(boxes) */
	/*
		// Turn our video frame into a a sprite to be drawn by Pixel
		pic := pixel.PictureDataFromImage(img)
		sprite := pixel.NewSprite(pic, pic.Bounds())

		// Clear any previous boxes
		imd.Clear()
		// Clear previous spires
		win.Clear(colornames.Black)
		// Draw the new frame first
		sprite.Draw(win, mat)

		// Draw a box around the objects
		curObj := 0
		// arbitrary detection threshold of 0.4
		for probabilities[curObj] > 0.4 {
			// box coordinates come in as [y1,x1,y2,x2]
			x1 := pic.Bounds().Max.X * float64(boxes[curObj][1])
			x2 := pic.Bounds().Max.X * float64(boxes[curObj][3])
			// TF (0,0) is the upper left, Pixel (0,0) is the lower left, so we need
			// to subtract the Y values from the max height so we draw from the bottom up
			y1 := pic.Bounds().Max.Y - (pic.Bounds().Max.Y * float64(boxes[curObj][0]))
			y2 := pic.Bounds().Max.Y - (pic.Bounds().Max.Y * float64(boxes[curObj][2]))

			objColor := colornames.Map[colornames.Names[int(classes[curObj])]]

			// Draw the box label
			txt := text.New(pixel.V(x1, y1), atlas)
			txt.Color = objColor
			txt.WriteString(getLabel(curObj, probabilities, classes))
			txt.Draw(win, pixel.IM.Scaled(txt.Orig, 2))

			// Push the box onto the draw stack
			imd.Color = objColor
			imd.Push(pixel.V(x1, y1), pixel.V(x2, y2))
			imd.Rectangle(1.0)

			curObj++*/
}

/*
		default:
		}

		// draw whatever's in the stack
		imd.Draw(win)
		win.Update()

		// calculate frame rate
		frames++
		select {
		case <-second:
			win.SetTitle(fmt.Sprintf("%s | FPS: %d", cfg.Title, frames))
			frames = 0
		default:
		}
	}
}
*/

// Build a graph to decode bitmap input into the proper tensor shape
// The object detection models take an input of [1,?,?,3]

func decodeBitmapGraph() (g *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	g, err = s.Finalize()
	return
}

/*
func constructGraphToNormalizeImage() (graph3 *tf.Graph, input3, output3 tf.Output, err error) {

	const (
		H, W  = 300, 300
		Mean  = float32(0)
		Scale = float32(1)
	)

	s := op.NewScope()
	input3 = op.Placeholder(s, tf.String)

	output3 = op.Div(s,
		op.ResizeBilinear(s,
			op.ExpandDims(s,
				op.Cast(s,
					op.DecodeJpeg(s, input3, op.DecodeJpegChannels(3)), tf.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{H, W})),
		op.Const(s.SubScope("scale"), Scale))

	graph3, err = s.Finalize()

	return graph3, input3, output3, err
}

*/
// Make a tensor from jpg image bytes
func makeTensorFromImage(filename string) (*tf.Tensor, error) {

	img, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Println("prob1-------------------------------------------------------------------------")
	}
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(img))
	if err != nil {
		return nil, err
	}
	// Creates a tensorflow graph to decode the jpeg image
	g, input, output, err := decodeBitmapGraph()
	if err != nil {
		return nil, err
	}
	// Execute that graph to decode this one image
	sess, err := tf.NewSession(g, nil)
	if err != nil {
		return nil, err
	}
	defer sess.Close()
	normalized, err := sess.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	/*
		r := bytes.NewReader(img)
		i, _, err := image.Decode(r)
		if err != nil {
			return nil, nil, err
		}
	*/
	return normalized[0], nil
}

/*
func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Println("prob1-------------------------------------------------------------------------")
	}
	fmt.Println("okayhere1----------------------------------")
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		fmt.Println("prob2")
	}
	fmt.Println("okayhere2----------------------------------")
	// Construct a graph to normalize the image
	graph2, input2, output2, err := constructGraphToNormalizeImage()
	if err != nil {
		fmt.Println("prob3")
	}
	fmt.Println("okayhere3----------------------------------")
	// Execute that graph to normalize this one image

	session, err := tf.NewSession(graph2, nil)
	if err != nil {
		fmt.Println("prob4")
	}
	defer session.Close()
	fmt.Println("okayhere4----------------------------------")
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input2: tensor},
		[]tf.Output{output2},
		nil)
	if err != nil {
		fmt.Println("prob5")
	}
	fmt.Println("okayhere5----------------------------------")
	return normalized[0], nil
}
*/
// Run the image through the model
/*
func predictObjectBoxes(input *tf.Tensor) (probabilities, classes []float32, boxes [][]float32) { //, numdetect float32, err error) {
	// Get all the input and output operations
	start := time.Now()
	fmt.Println("okayhere6----------------------------------")
	inputop := graph.Operation("image_tensor")
	fmt.Println("okayhere7----------------------------------")
	// Output ops
	o1 := graph.Operation("detection_boxes")
	o2 := graph.Operation("detection_scores")
	o3 := graph.Operation("detection_classes")
	o4 := graph.Operation("num_detections")
	fmt.Println("okayhere8----------------------------------")

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputop.Output(0): input,
		},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil)
	if err != nil {
		log.Fatalf("Error running session: %v", err)
	}
	elapsed := time.Since(start)
	log.Printf("Inference took %s", elapsed)
	fmt.Println("okayhere9----------------------------------")
	probabilities = output[1].Value().([][]float32)[0]
	classes = output[2].Value().([][]float32)[0]
	boxes = output[0].Value().([][][]float32)[0]
	//numdetect = output[3].Value().([]float32)[0]

	return
}
*/
