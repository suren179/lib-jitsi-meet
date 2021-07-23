/**
 * Create By: @ Shubham Joshi
 * Reference Links:-
 *      https://github.com/breizhn/DTLN/issues/4
 * This file loads dtln model from specified location and serves ti to html page for making predictions to denoiseoutput
 *
 * Usage:-
 *
 *      Step1: Import the script in html page:-
 *              <script src="dtln_model.js"></script>
 *
 *      Step2: Do the predictions using:-
 *             const predictedAudioData = await predict(<audio array>);
 */

// below blockLen and block_shift are to be changed based on the model
var blockLen = 1024;
var block_shift = 256;
var sampling_rate = 48000;

/**
 * This function is used to concat 2 javascript arrays
 * @param {*} first
 * @param {*} second
 * @returns
 */
function Float32Concat(first, second) {
    var firstLength = first.length,
        result = new Float32Array(firstLength + second.length);

    result.set(first);
    result.set(second, firstLength);

    return result;
}

/**
 * This class defines the overlap_and_add_layer
 */

class OverlapAddLayer extends tf.layers.Layer {
    static className = "OverlapAddLayer";
    constructor() {
        super({});
    }
    call(x) {
        result = tf.signal.overlap_and_add(x, OverlapAddLayer.block_shift);
        return result;
    }
}


/**
 * The below class implements STFT layer of DTLN Model
 * rEFERENCES: https://github.com/dntj/jsfft
 */
class STFTlayer extends tf.layers.Layer {
    static className = "STFTlayer";
    constructor() {
        console.log("STFTLayer Constructor");
        super({});
    }
    call(x) {
        console.log("Call Called for STFTLayer");
        console.log("***********************x:", x);
        // console.info("x[0].shape", x[0].shape);

        let reals = [];
        let imags = [];

        let frame_arr = null;
        let frame = null;
        for (let ii = 0; ii < x[0].shape[0]; ii++) {
            console.log("x[0]:", x[0])
            console.log("x[0].slice([ii],[0,x[0].shape[1]]).shape", x[0].slice([ii], [0, x[0].shape[1]]).shape)
            frame = tf.signal.frame(
                x[0].slice([ii, 0], [1, x[0].shape[1]]).flatten(),
                blockLen,
                block_shift
            );

            for (let ij = 0; ij < frame.shape[0]; ij++) {

                // const data = new ComplexArray(1024).map((value, i, n) => {
                //     value.real = frame_arr[i];
                // });

                // const frequencies = data.FFT();
                // reals.push(tf.tensor(frequencies.real.slice(0, 513)));
                // imags.push(tf.tensor(frequencies.imag.slice(0, 513)));
                frame_arr = frame.slice([ij, 0], [1, blockLen]).dataSync();
                var dft = new FFT(blockLen, sampling_rate);
                dft.forward(frame_arr);
                reals.push(tf.tensor(Float32Array.from(dft.real.slice(0, 513))));
                imags.push(tf.tensor(Float32Array.from(dft.imag.slice(0, 513))));
            }
        }

        let num_of_rec = x[0].shape[0];
        let len_of_ft = reals[0].shape[0];


        // Calculating Magnitudes of complex number
        // Reference: https://www.expii.com/t/magnitude-of-a-complex-number-4944
        //            https://gubner.ece.wisc.edu/notes/MagnitudeAndPhaseOfComplexNumbers.pdf
        //            https://github.com/corbanbrook/dsp.js



        reals = tf.concat(reals).flatten().dataSync();
        imags = tf.concat(imags).flatten().dataSync();

        let phase = reals.map(function (rel, i) {
            return Math.atan2(rel, imags[i]);
        })
        phase = tf.reshape(phase, [num_of_rec, frame.shape[0], len_of_ft]);

        let mag = tf.sqrt(tf.add(tf.square(imags), tf.square(reals)));
        mag = tf.reshape(mag, [num_of_rec, frame.shape[0], len_of_ft]);
        //let phase2 = tf.reshape(phase, [num_of_rec, 1, len_of_ft]);

        return [mag, phase];
    }

    computeOutputShape(inputShape) {
        console.log("Input Shape STFT:" + inputShape);
        return [
            [null, null, 513],
            [null, null, 513],
        ];
    }
}


/**
 * The below class implements Inverse fourier transformation layer
 * Reference: https://github.com/indutny/fft.js/
 */
class IFFTlayer extends tf.layers.Layer {
    static className = "IFFTlayer";
    constructor(config) {
        super(config);
    }
    call(x) {
        console.log("Calling IFFT:");
        // The below code converts magnitude and phase to complex number using polar coordinate approach
        // Refer the following link for more information , https://www.intmath.com/complex-numbers/convert-polar-rectangular-interactive.php
        let x_shape = x[0].shape;
        let real_arr = tf.mul(tf.cos(x[1]), tf.exp(x[0])).flatten().dataSync();
        let imag_arr = tf.mul(tf.sin(x[1]), tf.exp(x[0])).flatten().dataSync();
        //let cplx_arr = tf.stack([real_arr, imag_arr], 1).dataSync();

        let finalSignal = []

        for (let k = 0; k < x_shape[0]; k++) {
            for (let l = 0; l < x_shape[1]; l++) {

                // const data = new ComplexArray(513).map((value, i, n) => {
                //     if (i <= 513) {
                //         value.real = real_arr[i];
                //         value.imag = imag_arr[i];
                //     } else {
                //         value.real = real_arr[513 - (i - 513)];
                //         value.imag = -1*imag_arr[513 - (i - 513)];
                //     }
                // });

                // var out = data.InvFFT();

                var fft_len = (blockLen / 2) + 1;
                var real_sig = real_arr.slice((k + l) * fft_len, (k + l) * fft_len + fft_len);
                var img_sig = imag_arr.slice((k + l) * fft_len, (k + l) * fft_len + fft_len);
                real_sig = Float32Concat(real_sig, real_sig.reverse().slice(0, fft_len - 1));
                img_sig = Float32Concat(img_sig, img_sig.reverse().slice(0, fft_len - 1));

                var dft_inv = new FFT(blockLen, sampling_rate);
                var ifft_arr = dft_inv.inverse(real_sig, img_sig);
                console.log("inv_dft.inverse:", ifft_arr);

                finalSignal.push(tf.tensor(Float32Array.from(ifft_arr)));
            }
        }
        let flat_opt = tf.concat(finalSignal);
        return tf.reshape(flat_opt, [x_shape[0], x_shape[1], blockLen]);
    }

    static get className() {
        console.log(className);
        return className;
    }

    computeOutputShape(inputShape) {
        return [null, null, blockLen];
    }
}

/**
 * The below class implements add and log layer of DTLN
 */
class AddAndLog extends tf.layers.Layer {
    static className = "AddAndLog";
    constructor(config) {
        super(config);
    }
    call(x) {
        x = x[0]
        return tf.log(tf.add(x, tf.tensor(1e-7)));
    }
    static get className() {
        return className;
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
}

/**
 * The below class implements instance normalization layer of DTLN
 */
class InstantLayerNormalization extends tf.layers.Layer {
    static className = "InstantLayerNormalization";
    epsilon = 1e-7;
    gamma;
    beta;
    gamma_fill;
    beta_fill;
    constructor(config) {
        super(config);
    }
    getConfig() {
        const config = super.getConfig();
        return config;
    }

    build(input_shape) {
        this.gamma = this.addWeight(
            "gamma",
            [input_shape[2]],
            "float32",
            "ones"
        );

        this.beta = this.addWeight(
            "beta",
            [input_shape[2]],
            "float32",
            "ones"
        );
    }

    call(inputs) {
        inputs = inputs[0]

        var mean = tf.mean(inputs, -1, true);
        var variance = tf.mean(tf.square(tf.sub(inputs, mean)), -1, true);
        var std = tf.sqrt(tf.add(variance, this.epsilon));
        var outputs = tf.div(tf.sub(inputs, mean), std);
        var outputs = tf.mul(outputs, this.gamma.read());
        var outputs = tf.add(outputs, this.beta.read());
        return outputs;
    }

    computeOutputShape(inputShape) {
        return inputShape;
    }

    static get className() {
        console.log(className);
        return className;
    }
}

// Register all custom layers with tensorflow.js
tf.serialization.registerClass(IFFTlayer);
tf.serialization.registerClass(InstantLayerNormalization);
tf.serialization.registerClass(STFTlayer);
tf.serialization.registerClass(OverlapAddLayer);
tf.serialization.registerClass(AddAndLog);




/**
 * The below function loads the DTLN model from specified path and returns a model object
 * @returns model
 */
export async function loadDTLN_model(path) {
    console.log("Loading DTLN Model....");
    var model = await tf.loadLayersModel(path);
    console.log("DTLN Model loaded....");
    model = tf.model({ inputs: model.inputs, outputs: model.layers[model.layers.length - 2].output });
    return model;
}


// Initialize buffers to hold audio data
let in_buffer = new Float32Array(blockLen).fill(0);
let out_buffer = new Float32Array(blockLen).fill(0);
let out_file = new Float32Array(0).fill(0);



/**
 * This function is used to denoise audio.
 * This should only be used with realtime prediction streaming.
 * This function takes audio sample blocks strictly of size 256.
 * @ImportantNote : This function will return an array of '0' for first 4 block (each of size 256 samples),
 *                  and from 5th block it will return the denoised output of the previous 1st block.
 * @param {input is the audio array of strictly in multiples of `1024`} input
 * @returns denoised audio array 4 blocks prioor to current block
 */
export async function predict(input, model) {
  let output_result = new Float32Array(0);
  let num_frames = input.length / block_shift;

  for (let fi = 0; fi < num_frames - 4; fi = fi + 4) {
    let input_pred_2d = [];
    for (let al = 0; al < 4; al++) {
      const start_index = (fi + al) * block_shift;
      const end_index = (fi + al) * block_shift + block_shift;
      in_buffer = Float32Concat(
        in_buffer.slice(end_index - start_index, in_buffer.length),
        input.slice(start_index, end_index)
      );
      input_pred_2d.push(in_buffer);
    }

    let predictions = await model.predict(tf.tensor2d(input_pred_2d)).data();

    for (let al = 0; al < 4; al++) {
      out_buffer = Float32Concat(
        out_buffer.slice(block_shift, out_buffer.length),
        new Float32Array(block_shift).fill(0)
      );
      out_buffer = out_buffer.map((a, i) => a + predictions[al * blockLen + i]);
      output_result = Float32Concat(
        output_result,
        out_buffer.slice(0, block_shift)
      );
    }
  }
  return output_result;
}

/**
 * This function gets called on windows.onload function and loads dtln model
 * @param {relative path of the tensorflow js model} path
 */
function load_model(path) {
    loadDTLN_model(path).then(
        function (model_ret) {
            model = model_ret;
            model.summary();
            console.log("Model Loaded !!!!!");
        },
        function (finalResult) {
            console.log("Error Loaded !!!!!" + finalResult);
        }
    );
}

/**
 * This method willbe used to debug the model at layer by layer level
 * @param {*} inp
 * @param {*} model
 * @returns
 */
async function predictAudioSuperSync(inp, model) {
    // sftp

    console.log(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;", inp);
    var d1 = new Date();
    var result = await model.layers[1].apply([inp]);
    var d2 = new Date();
    var diff = d2.getTime() - d1.getTime();
    console.info("model.layers[1]", diff);

    var mag = await result[0].data();
    console.log("mag array:", mag);
    mag = tf.expandDims(tf.expandDims(tf.tensor(mag), 0), 0);

    var angle = await result[1].data();
    console.log("angle array:", angle);
    angle = tf.expandDims(tf.expandDims(tf.tensor(angle), 0), 0);

    console.log("mag:", mag);
    console.log("mag shape:", mag.shape);
    console.log("angle:", angle);

    // add and mul
    d1 = new Date();
    var addAndMul = await model.layers[2].apply([mag]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[2]", diff);
    addAndMul = tf.expandDims(tf.expandDims(tf.tensor(addAndMul), 0), 0);
    console.log("Outer addAndMul:", addAndMul);
    console.log("Outer addAndMul shape:", addAndMul.shape);

    // instanceNorm layer
    d1 = new Date();
    var mag_norm = await model.layers[3].apply([addAndMul]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[3]", diff);
    console.log("Outer mag_norm before expand dims:", mag_norm);
    mag_norm = tf.expandDims(tf.expandDims(tf.tensor(mag_norm), 0), 0);
    console.log("Outer mag_norm:", mag_norm);
    console.log("Outer mag_norm shape:", mag_norm.shape);

    // ************************ Seperation kernal start ****************************************
    // lstm1 layer
    d1 = new Date();
    var x = await model.layers[4].apply([mag_norm]).data();
    // var mag_norm = await model.layers[4].apply(addAndMul).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[4]", diff);
    x = tf.expandDims(tf.expandDims(tf.tensor(x), 0), 0);

    // Dropout
    d1 = new Date();
    x = await model.layers[5].apply([x]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[5]", diff);
    x = tf.expandDims(tf.expandDims(tf.tensor(x), 0), 0);

    // lstm2 layer
    d1 = new Date();
    x = await model.layers[6].apply([x]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[6]", diff);
    x = tf.expandDims(tf.expandDims(tf.tensor([x]), 0), 0);

    // creating the mask with a Dense and an Activation layer
    d1 = new Date();
    var mask = await model.layers[7].apply([x]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[7]", diff);
    mask = tf.expandDims(tf.expandDims(tf.tensor(mask), 0), 0);

    d1 = new Date();
    var mask_1 = await model.layers[8].apply([mask]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[8]", diff);
    mask_1 = tf.expandDims(tf.expandDims(tf.tensor(mask_1), 0), 0);
    // ************************ Seperation kernal end ****************************************

    // Multiply
    d1 = new Date();
    var estimated_mag = await model.layers[9].apply([mag, mask_1]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[9]", diff);
    console.log(
        "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLayer9:",
        estimated_mag
    );
    estimated_mag = tf.expandDims(tf.expandDims(tf.tensor(estimated_mag), 0), 0);

    // IIFT transform frames back to time domain
    d1 = new Date();
    var estimated_frames_1 = await model.layers[10]
        .apply([estimated_mag, angle])
        .data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[10]", diff);
    estimated_frames_1 = tf.expandDims(
        tf.expandDims(tf.tensor(estimated_frames_1), 0),
        0
    );

    // Conv1d encode time domain frames to feature domain
    d1 = new Date();
    var encoded_frames = await model.layers[11].apply([estimated_frames_1]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[11]", diff);
    encoded_frames = tf.expandDims(
        tf.expandDims(tf.tensor(encoded_frames), 0),
        0
    );

    //Instance normalize the input to the separation kernel
    d1 = new Date();
    var encoded_frames_norm = await model.layers[12].apply([encoded_frames]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[12]", diff);
    encoded_frames_norm = tf.expandDims(
        tf.expandDims(tf.tensor(encoded_frames_norm), 0),
        0
    );

    // ************************ Seperation kernal start ****************************************
    // lstm1 layer
    d1 = new Date();
    x = await model.layers[13].apply([encoded_frames_norm]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[13]", diff);
    x = tf.expandDims(tf.expandDims(tf.tensor(x), 0), 0);

    // Dropout
    d1 = new Date();
    x = await model.layers[14].apply([x]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[14]", diff);
    x = tf.expandDims(tf.expandDims(tf.tensor(x), 0), 0);

    // lstm2 layer
    d1 = new Date();
    x = await model.layers[15].apply([x]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[15]", diff);
    x = tf.expandDims(tf.expandDims(tf.tensor(x), 0), 0);

    // creating the mask with a Dense and an Activation layer
    d1 = new Date();
    mask = await model.layers[16].apply([x]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[16]", diff);
    mask = tf.expandDims(tf.expandDims(tf.tensor([mask]), 0), 0);

    d1 = new Date();
    var mask_2 = await model.layers[17].apply([mask]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[17]", diff);
    mask_2 = tf.expandDims(tf.expandDims(tf.tensor([mask_2]), 0), 0);
    // ************************ Seperation kernal end ****************************************

    // Multiplymultiply encoded frames with the mask
    d1 = new Date();
    var estimated = await model.layers[18].apply([encoded_frames, mask_2]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[18]", diff);
    estimated = tf.expandDims(tf.expandDims(tf.tensor(estimated), 0), 0);

    // Conv1D decode the frames back to time domain
    d1 = new Date();
    var decoded_frames = await model.layers[19].apply([estimated]).data();
    d2 = new Date();
    diff = d2.getTime() - d1.getTime();
    console.info("model.layers[19]", diff);
    //decoded_frames=tf.tensor(decoded_frames)
    // decoded_frames=tf.expandDims(tf.expandDims(tf.tensor(decoded_frames),0),0);

    // //OverlapAndAdd create waveform with overlap and add procedure
    // estimated_sig =await  model.layers[20].apply(decoded_frames).data();
    // estimated_sig=tf.expandDims(tf.expandDims(tf.tensor(estimated_sig),0),0);

    // alert(
    //   "!!!!!!!!!!!! Prediction Complete !!!!!!!!!!!!!!!!!!",
    //   decoded_frames
    // );

    console.log(
        "!!!!!!!!!!!! Prediction Complete !!!!!!!!!!!!!!!!!!",
        decoded_frames
    );
    return decoded_frames;
}

