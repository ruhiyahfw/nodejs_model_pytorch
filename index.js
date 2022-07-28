const tf = require('@tensorflow/tfjs-node');
const fs = require('fs')
const ort = require('onnxruntime-node');
var Jimp = require('jimp');

const encodeBase64 = (photoPath) => {
    var imageAsBase64 = fs.readFileSync(photoPath, 'base64');
    return imageAsBase64;
}

async function main(){
    try{
        // create a new session and load the specific model.
        const session = await ort.InferenceSession.create('./model_ignore/80_pesen_fit_base_patch8_224.onnx');
        console.log("berhasil buka model")

        // Load image.
        const base64image = encodeBase64("./foto/Abyssinian.jpg")
        const imageBuffer = Buffer.from(base64image, 'base64')
        console.log(imageBuffer)

        let image = tf.node.decodeImage(imageBuffer);
        console.log(image.shape)
        image = tf.image.resizeBilinear(image, size = [224, 224]); 
        image = image.cast('float32').div(255);
        // resize the image
        image = image.expandDims(); // to add the most left axis of size 1
        const b = image.shape
        console.log("lala")
        console.log(b)

        // get the tensor
        const t = tf.node.decodeImage(image)
        console.log(t)

        // feed inputs and run
        const feeds = {image_1_3_224_224: t}
        const results = await session.run(feeds);

        // read from results
        const dataC = results.c.data;
        console.log(`data of result tensor 'c': ${dataC}`);

        
    }
    catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main()