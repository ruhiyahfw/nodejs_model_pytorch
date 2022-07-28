const tf = require('@tensorflow/tfjs-node');
const fs = require('fs')
const ort = require('onnxruntime-node');
var Jimp = require('jimp');


labels = ['Abyssinian', 'American Shorthair', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Burmilla', 'Chartreux', 'Cornish Rex', 'Devon Rex', 'Domestic', 'Egyptian Mau', 'Exotic', 'Havana Brown', 'Himalaya', 'Japanese Bobtail', 'Khao Manee', 'Korat', 'LaPerm', 'Lykoi', 'Maine Coon', 'Manx', 'Norwegian Forest', 'Ocicat', 'Oriental', 'Persian', 'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Singapura', 'Snowshoe', 'Somali', 'Sphynx', 'Toybob', 'Turkish Angora', 'Turkish Van']

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
        const base64image = encodeBase64("./foto/LaPerm.jpg")
        const imageBuffer = Buffer.from(base64image, 'base64')
        console.log(imageBuffer)

        let image = tf.node.decodeImage(imageBuffer);

        image = tf.image.resizeBilinear(image, size = [224, 224]); 
        image = image.cast('float32').div(255);

        image = tf.transpose(image, perm=[2,0,1])

        // resize the image
        image = image.expandDims(); // to add the most left axis of size 1
        const b = image.dtype
        
        console.log(b)

        buffer = image.dataSync().buffer
        t = new Float32Array(buffer)
        const tensorA = new ort.Tensor('float32', t, [1, 3, 224, 224]);

        // feed inputs and run
        const feeds = {image_1_3_224_224:tensorA}
        const results = await session.run(feeds);

        console.log("===================")
        probArr = results.breeds.data

        const maxProb = Math.max(...probArr)
        const idxBreed = probArr.indexOf(maxProb)

        console.log(`breed terdeteksi = ${labels[idxBreed]} dengan probability ${maxProb}`)

        // // read from results
        // const dataC = results.c.data;
        // console.log(`data of result tensor 'c': ${dataC}`);

        
    }
    catch (e) {
        console.error(`${e}.`);
    }
}

main()