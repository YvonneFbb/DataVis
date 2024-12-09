import objc
from AppKit import NSData, NSImage
from CoreFoundation import (
    CFRunLoopRunInMode,
    kCFRunLoopDefaultMode,
    CFRunLoopStop,
    CFRunLoopGetCurrent,
)
from PIL import ImageDraw

class AppleLiveTextOCR:
    def __init__(self, locales=["zh"]):
        self.locales = locales
        self.load_register_framework()

    def load_register_framework(self):
        # Load VisionKit library
        objc.loadBundle(
            "VisionKit", globals(), "/System/Library/Frameworks/VisionKit.framework"
        )

        # Register necessary metadata
        objc.registerMetaDataForSelector(
            b"VKCImageAnalyzer",
            b"processRequest:progressHandler:completionHandler:",
            {
                "arguments": {
                    3: {
                        "callable": {
                            "retval": {"type": b"v"},
                            "arguments": {
                                0: {"type": b"^v"},
                                1: {"type": b"d"},
                            },
                        }
                    },
                    4: {
                        "callable": {
                            "retval": {"type": b"v"},
                            "arguments": {
                                0: {"type": b"^v"},
                                1: {"type": b"@"},
                                2: {"type": b"@"},
                            },
                        }
                    },
                }
            },
        )

    def pil_image_to_ns_image(self, img):
        import io

        image_bytes = io.BytesIO()
        img.save(image_bytes, format="TIFF")
        ns_data = NSData.dataWithBytes_length_(
            image_bytes.getvalue(), len(image_bytes.getvalue())
        )
        ns_image = NSImage.alloc().initWithData_(ns_data)
        return ns_image

    def perform_ocr(self, img):
        ns_image = self.pil_image_to_ns_image(img)
        analyzer = objc.lookUpClass("VKCImageAnalyzer").alloc().init()
        request = (
            objc.lookUpClass("VKCImageAnalyzerRequest")
            .alloc()
            .initWithImage_requestType_(ns_image, 1)
        )  # VKAnalysisTypeText
        request.setLocales_(self.locales)  # Set recognition languages

        result = []

        def process_handler(analysis, error):
            if error:
                result.append({"text": "Error: " + str(error), "bounds": None})
            else:
                lines = analysis.allLines()
                if lines:
                    for line in lines:
                        for x in line.children():
                            char_text = x.string()
                            bounding_box = x.quad().boundingBox()
                            # Extracting the origin and size from the bounding box
                            bounds = {
                                "origin": (bounding_box.origin.x, bounding_box.origin.y),
                                "size": (bounding_box.size.width, bounding_box.size.height),
                            }
                            result.append({"text": char_text, "bounds": bounds})
            CFRunLoopStop(CFRunLoopGetCurrent())

        analyzer.processRequest_progressHandler_completionHandler_(
            request, lambda progress: None, process_handler
        )

        # Run the run loop to wait for the OCR to finish
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 10.0, False)

        return result

    def draw_bounding_boxes(self, img, characters):
        draw = ImageDraw.Draw(img.copy())
        img_width, img_height = img.size

        for char_info in characters:
            if char_info["bounds"]:
                bounds = char_info["bounds"]
                # Convert normalized coordinates to pixel coordinates
                x = bounds["origin"][0] * img_width
                y = bounds["origin"][1] * img_height
                width = bounds["size"][0] * img_width
                height = bounds["size"][1] * img_height
                # Draw rectangle around the character
                draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

        return img
