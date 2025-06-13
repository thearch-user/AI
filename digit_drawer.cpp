#include <SDL2/SDL.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <random>

// Constants
const int WINDOW_WIDTH = 280;
const int WINDOW_HEIGHT = 280;
const int PIXEL_SIZE = 10;
const int CANVAS_WIDTH = 28;
const int CANVAS_HEIGHT = 28;
const int HIDDEN_SIZE = 128;
const int OUTPUT_SIZE = 10;

// Neural Network Class
class NeuralNetwork {
private:
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<double> bias_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> bias_output;

    double leaky_relu(double x, double alpha = 0.01) {
        return x > 0 ? x : alpha * x;
    }

    double leaky_relu_derivative(double x, double alpha = 0.01) {
        return x > 0 ? 1 : alpha;
    }

    void softmax(std::vector<double>& z) {
        double max_z = *std::max_element(z.begin(), z.end());
        double sum = 0.0;
        
        for (auto& val : z) {
            val = std::exp(val - max_z);
            sum += val;
        }
        
        for (auto& val : z) {
            val /= sum;
        }
    }

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size) {
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / input_size));
        
        // Initialize weights and biases
        weights_input_hidden.resize(hidden_size, std::vector<double>(input_size));
        for (auto& row : weights_input_hidden) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
        
        bias_hidden.resize(hidden_size, 0.0);
        
        dist = std::normal_distribution<double>(0.0, std::sqrt(2.0 / hidden_size));
        weights_hidden_output.resize(output_size, std::vector<double>(hidden_size));
        for (auto& row : weights_hidden_output) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
        
        bias_output.resize(output_size, 0.0);
    }

    void load_weights(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open weights file: " << filename << std::endl;
            return;
        }

        // Load weights and biases from file
        // Format: weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
        // Implementation depends on how you saved the weights in Python
        // This is just a placeholder
    }

    std::vector<double> forward(const std::vector<double>& input) {
        // Hidden layer
        std::vector<double> hidden(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            for (int j = 0; j < input.size(); ++j) {
                hidden[i] += weights_input_hidden[i][j] * input[j];
            }
            hidden[i] += bias_hidden[i];
            hidden[i] = leaky_relu(hidden[i]);
        }

        // Output layer
        std::vector<double> output(OUTPUT_SIZE, 0.0);
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                output[i] += weights_hidden_output[i][j] * hidden[j];
            }
            output[i] += bias_output[i];
        }

        softmax(output);
        return output;
    }

    int predict(const std::vector<double>& input) {
        auto output = forward(input);
        return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    }
};

// Drawing Application
class DigitDrawer {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    std::vector<std::vector<bool>> canvas;
    NeuralNetwork nn;

    void clear_canvas() {
        for (auto& row : canvas) {
            std::fill(row.begin(), row.end(), false);
        }
    }

    void draw_point(int x, int y) {
        // Draw a 5x5 square around the point
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < WINDOW_WIDTH && ny >= 0 && ny < WINDOW_HEIGHT) {
                    canvas[ny][nx] = true;
                }
            }
        }
    }

    std::vector<double> get_downsampled_image() {
        std::vector<double> small_image(CANVAS_WIDTH * CANVAS_HEIGHT, 0.0);
        
        double scale_x = static_cast<double>(WINDOW_WIDTH) / CANVAS_WIDTH;
        double scale_y = static_cast<double>(WINDOW_HEIGHT) / CANVAS_HEIGHT;
        
        for (int y = 0; y < CANVAS_HEIGHT; ++y) {
            for (int x = 0; x < CANVAS_WIDTH; ++x) {
                int start_x = static_cast<int>(x * scale_x);
                int start_y = static_cast<int>(y * scale_y);
                int end_x = static_cast<int>((x + 1) * scale_x);
                int end_y = static_cast<int>((y + 1) * scale_y);
                
                int count = 0;
                for (int sy = start_y; sy < end_y; ++sy) {
                    for (int sx = start_x; sx < end_x; ++sx) {
                        if (canvas[sy][sx]) {
                            count++;
                        }
                    }
                }
                
                small_image[y * CANVAS_WIDTH + x] = static_cast<double>(count) / ((end_x - start_x) * (end_y - start_y));
            }
        }
        
        return small_image;
    }

public:
    DigitDrawer() : nn(CANVAS_WIDTH * CANVAS_HEIGHT, HIDDEN_SIZE, OUTPUT_SIZE) {
        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("Digit Drawer", 
                                SDL_WINDOWPOS_CENTERED, 
                                SDL_WINDOWPOS_CENTERED,
                                WINDOW_WIDTH, WINDOW_HEIGHT, 
                                SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        
        // Initialize canvas
        canvas.resize(WINDOW_HEIGHT, std::vector<bool>(WINDOW_WIDTH, false));
        
        // Load pre-trained weights (you would need to implement this)
        // nn.load_weights("weights.txt");
    }

    ~DigitDrawer() {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void run() {
        bool running = true;
        bool drawing = false;
        SDL_Event event;
        
        clear_canvas();
        
        while (running) {
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT:
                        running = false;
                        break;
                        
                    case SDL_MOUSEBUTTONDOWN:
                        if (event.button.button == SDL_BUTTON_LEFT) {
                            drawing = true;
                            draw_point(event.button.x, event.button.y);
                        }
                        break;
                        
                    case SDL_MOUSEBUTTONUP:
                        if (event.button.button == SDL_BUTTON_LEFT) {
                            drawing = false;
                            
                            // Get downsampled image
                            auto small_image = get_downsampled_image();
                            
                            // Predict
                            int prediction = nn.predict(small_image);
                            std::cout << "Predicted digit: " << prediction << std::endl;
                        }
                        break;
                        
                    case SDL_MOUSEMOTION:
                        if (drawing) {
                            draw_point(event.motion.x, event.motion.y);
                        }
                        break;
                        
                    case SDL_KEYDOWN:
                        if (event.key.keysym.sym == SDLK_c) {
                            clear_canvas();
                        }
                        break;
                }
            }
            
            // Render
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            
            // Draw canvas
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            for (int y = 0; y < WINDOW_HEIGHT; ++y) {
                for (int x = 0; x < WINDOW_WIDTH; ++x) {
                    if (canvas[y][x]) {
                        SDL_RenderDrawPoint(renderer, x, y);
                    }
                }
            }
            
            SDL_RenderPresent(renderer);
        }
    }
};

int main(int argc, char* argv[]) {
    DigitDrawer app;
    app.run();
    return 0;
}
