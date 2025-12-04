#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <iomanip>
#include <map>

// Structure to hold decoded pixel information
struct DecodedPixel {
    uint8_t class_id;
    uint8_t confidence_raw;
    float confidence_normalized;
};

class ImageDecoder {
private:
    std::vector<uint16_t> encoded_data;
    size_t width;
    size_t height;

public:
    ImageDecoder(size_t w, size_t h) : width(w), height(h) {
        encoded_data.resize(w * h);
    }

    // Load encoded data from binary file
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }

        file.read(reinterpret_cast<char*>(encoded_data.data()), 
                  encoded_data.size() * sizeof(uint16_t));
        
        if (!file) {
            std::cerr << "Error: Could not read expected amount of data" << std::endl;
            return false;
        }

        file.close();
        return true;
    }

    // Manually set encoded data (for testing)
    void setEncodedData(const std::vector<uint16_t>& data) {
        if (data.size() == encoded_data.size()) {
            encoded_data = data;
        } else {
            std::cerr << "Error: Data size mismatch" << std::endl;
        }
    }

    // Decode a single 16-bit value
    DecodedPixel decodePixel(uint16_t encoded_value) const {
        DecodedPixel pixel;
        
        // Extract upper 8 bits for class_id
        pixel.class_id = static_cast<uint8_t>(encoded_value >> 8);
        
        // Extract lower 8 bits for confidence
        pixel.confidence_raw = static_cast<uint8_t>(encoded_value & 0xFF);
        
        // Normalize to 0.0-1.0
        pixel.confidence_normalized = pixel.confidence_raw / 255.0f;
        
        return pixel;
    }

    // Decode entire image
    std::vector<DecodedPixel> decodeImage() const {
        std::vector<DecodedPixel> decoded;
        decoded.reserve(encoded_data.size());

        for (size_t i = 0; i < encoded_data.size(); ++i) {
            decoded.push_back(decodePixel(encoded_data[i]));
        }

        return decoded;
    }

    // Get statistics per class
    std::map<uint8_t, std::vector<uint8_t>> getClassConfidences() const {
        std::map<uint8_t, std::vector<uint8_t>> class_confidences;

        for (size_t i = 0; i < encoded_data.size(); ++i) {
            uint16_t encoded_value = encoded_data[i];
            uint8_t class_id = static_cast<uint8_t>(encoded_value >> 8);
            uint8_t confidence_byte = static_cast<uint8_t>(encoded_value & 0xFF);
            
            class_confidences[class_id].push_back(confidence_byte);
        }

        return class_confidences;
    }

    // Print decoded image
    void printDecoded(size_t max_rows = 10, size_t max_cols = 10) const {
        std::cout << "\nDecoded Image (showing up to " << max_rows << "x" << max_cols << "):\n";
        std::cout << std::string(60, '-') << std::endl;

        size_t rows_to_show = std::min(height, max_rows);
        size_t cols_to_show = std::min(width, max_cols);

        for (size_t y = 0; y < rows_to_show; ++y) {
            for (size_t x = 0; x < cols_to_show; ++x) {
                size_t idx = y * width + x;
                DecodedPixel pixel = decodePixel(encoded_data[idx]);
                
                std::cout << "C" << std::setw(2) << (int)pixel.class_id 
                          << ":" << std::setw(3) << (int)pixel.confidence_raw << " ";
            }
            if (width > cols_to_show) {
                std::cout << "...";
            }
            std::cout << std::endl;
        }
        
        if (height > rows_to_show) {
            std::cout << "..." << std::endl;
        }
    }

    // Print statistics
    void printStatistics() const {
        auto class_confidences = getClassConfidences();
        
        std::cout << "\nClass Statistics:\n";
        std::cout << std::string(60, '-') << std::endl;
        
        for (const auto& pair : class_confidences) {
            uint8_t class_id = pair.first;
            const auto& confidences = pair.second;
            
            // Calculate average confidence
            uint64_t sum = 0;
            for (uint8_t conf : confidences) {
                sum += conf;
            }
            float avg_confidence = sum / (float)confidences.size();
            
            std::cout << "Class " << std::setw(3) << (int)class_id 
                      << ": Count=" << std::setw(6) << confidences.size()
                      << ", Avg Confidence=" << std::setw(6) << std::fixed 
                      << std::setprecision(2) << avg_confidence << "/255 ("
                      << std::setprecision(1) << (avg_confidence/255.0f * 100.0f) << "%)"
                      << std::endl;
        }
    }

    // Save decoded image to CSV
    bool saveToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not create file " << filename << std::endl;
            return false;
        }

        file << "x,y,class_id,confidence_raw,confidence_normalized\n";

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                DecodedPixel pixel = decodePixel(encoded_data[idx]);
                
                file << x << "," << y << "," 
                     << (int)pixel.class_id << ","
                     << (int)pixel.confidence_raw << ","
                     << pixel.confidence_normalized << "\n";
            }
        }

        file.close();
        std::cout << "Saved decoded data to " << filename << std::endl;
        return true;
    }

    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }
};

int main(int argc, char* argv[]) {
    std::cout << "16-bit Image Decoder\n";
    std::cout << "Format: [8 bits class_id][8 bits confidence]\n";
    std::cout << std::string(60, '=') << std::endl;

    // Example 1: Test with your specific value
    std::cout << "\nTest Case: Binary 00000010 10110011\n";
    ImageDecoder test_decoder(1, 1);
    std::vector<uint16_t> test_data = {0b0000001010110011}; // = 0x02B3
    test_decoder.setEncodedData(test_data);
    
    DecodedPixel test_pixel = test_decoder.decodePixel(test_data[0]);
    std::cout << "Encoded value: 0x" << std::hex << std::setw(4) << std::setfill('0') 
              << test_data[0] << std::dec << std::setfill(' ') << std::endl;
    std::cout << "Class ID: " << (int)test_pixel.class_id << std::endl;
    std::cout << "Confidence (raw): " << (int)test_pixel.confidence_raw << std::endl;
    std::cout << "Confidence (normalized): " << std::fixed << std::setprecision(4) 
              << test_pixel.confidence_normalized << std::endl;

    // Example 2: Load from file if provided
    if (argc > 1) {
        std::string input_file = argv[1];
        size_t width = 640;  // Default width
        size_t height = 480; // Default height

        if (argc > 3) {
            width = std::stoul(argv[2]);
            height = std::stoul(argv[3]);
        }

        std::cout << "\n\nLoading image from file: " << input_file << std::endl;
        std::cout << "Dimensions: " << width << "x" << height << std::endl;

        ImageDecoder decoder(width, height);
        
        if (decoder.loadFromFile(input_file)) {
            decoder.printDecoded(10, 10);
            decoder.printStatistics();
            
            // Save to CSV
            std::string output_csv = "decoded_output.csv";
            decoder.saveToCSV(output_csv);
        }
    } else {
        std::cout << "\n\nUsage: " << argv[0] << " <input_file> [width] [height]\n";
        std::cout << "Example: " << argv[0] << " encoded_image.bin 640 480\n";
    }

    return 0;
}