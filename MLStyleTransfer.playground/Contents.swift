//: MLStyleTransfer Playground
//: A Playground for testing your image stylizing CoreML model
//: It loads and compiles the model in runtime
//: You need to set the "Playground parameters" below
//: Please create a folder named "Shared Playground Data" in your ~/Documents folder
//: Enjoy your stylez :)
//: Doron Adler, @norod78

import AppKit
import PlaygroundSupport
import CoreML

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                      Playground parameters
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let numStyles  = 9                                              // Number of styles available in the model
let styleIdx = 0                                                // 0 <= styleIdx < numStyles // That is how you choose the style you want to apply
let modelInputSize = CGSize(width:768, height:768)              // You can try increasing this for better resolution but slower processing time
let inputImageFullPath = String("/Users/dadler/Projects/GitHub/MLStyleTransferPlayground/IMG_5176.jpg") // Full path of the image you wish to stylize
let mlModelFullPath = String("/Users/dadler/Projects/GitHub/MLStyleTransferPlayground/MyStyleTransfer.mlmodel") // Full path of the CoreML model file
let mlModelInputFeatureNameForImage  = String("image")           // The model's input feature name for the input image
let mlModelInputFeatureNameForIndex  = String("index")           // The model's input feature name for the selected style index
let mlModelOutputFeatureNameForImage = String("stylizedImage")   // The model's output feature name for the stylized image

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Please create a folder named "Shared Playground Data" in your ~/Documents folder
//                                       That is where the output images will be written to
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class StyleTransferInput : MLFeatureProvider {
    
    var input: CVPixelBuffer
    
    let styleArray = try? MLMultiArray(shape: [numStyles] as [NSNumber], dataType: MLMultiArrayDataType.double)
    
    var featureNames: Set<String> {
        get {
            return [mlModelInputFeatureNameForImage,mlModelInputFeatureNameForIndex]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == mlModelInputFeatureNameForImage) // The model's input feature name for "image"
        {
            return MLFeatureValue(pixelBuffer: input)
        }
        if (featureName == mlModelInputFeatureNameForIndex) // The model's input feature name for "index"
        {
            // Reset the index array
            for i in 0...((styleArray!.count)-1)
            {
                styleArray?[i] = 0.0
            }
            styleArray![styleIdx] = 1.0 // Set the index of the style you selected in the parameters above
            return MLFeatureValue(multiArray: styleArray!)
        }
        return nil
    }
    
    init(input: CVPixelBuffer) {
        self.input = input
    }
}

// Load the image
let imageOpt = NSImage(contentsOfFile: inputImageFullPath)
guard let image = imageOpt else {fatalError()}

// Setup some playground views
let imageView = NSImageView(image: image)
let stackView = NSStackView(views: [imageView])
stackView.orientation = .vertical
stackView.frame = CGRect(origin: .zero, size: image.size)
PlaygroundPage.current.liveView = stackView

let modelUrl = URL(fileURLWithPath:mlModelFullPath)
let compiledUrl = try MLModel.compileModel(at: modelUrl)
print("compiledUrl = \(compiledUrl)", compiledUrl)
let model = try MLModel(contentsOf: compiledUrl)

// set input size of the model
var modelInputRect = CGRect(origin: .zero, size: modelInputSize)

if let cgImage = image.cgImage(forProposedRect: &modelInputRect, context: nil, hints: nil) {
    let ciImage = CIImage(cgImage: cgImage)
    
    // create a cvpixel buffer
    var pixelBuffer: CVPixelBuffer?
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                 kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    CVPixelBufferCreate(kCFAllocatorDefault,
                        Int(modelInputSize.width),
                        Int(modelInputSize.height),
                        kCVPixelFormatType_32BGRA,
                        attrs,
                        &pixelBuffer)
    
    // put bytes into pixelBuffer
    let context = CIContext()
    context.render(ciImage, to: pixelBuffer!)
    
    // stylize the input image
    let input = StyleTransferInput(input:pixelBuffer!)
    let outFeatures = try model.prediction(from: input)
    
    // get the stylized output image
    if let featureValue = outFeatures.featureValue(for: mlModelOutputFeatureNameForImage) {
        if let output = featureValue.imageBufferValue {
            let outputImage = CIImage(cvPixelBuffer: output)
            let rep = NSCIImageRep(ciImage: outputImage)
            let nsImage = NSImage(size: rep.size)
            nsImage.addRepresentation(rep)
            let stackImageView = NSImageView(image: nsImage)
            stackView.addView(stackImageView, in: NSStackView.Gravity.top)
            
            let filepath = playgroundSharedDataDirectory.appendingPathComponent("export_\(styleIdx).tiff")
            if let imagedata = nsImage.tiffRepresentation(using: NSBitmapImageRep.TIFFCompression.none, factor: 0.8) {
                let fileURL = URL.init(fileURLWithPath: filepath.path)
                try! imagedata.write(to: fileURL, options: Data.WritingOptions.atomicWrite)
                print("Wrote \(imagedata) to \(filepath)")
            }
        }
    }
} else {
    fatalError()
}


