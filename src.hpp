#ifndef SRC_HPP
#define SRC_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>

typedef std::vector<std::vector<double>> IMAGE_T;

// Helper function to count connected components (holes)
int countHoles(const IMAGE_T &img) {
    int h = img.size();
    int w = img[0].size();
    std::vector<std::vector<bool>> visited(h, std::vector<bool>(w, false));
    
    auto floodFill = [&](int y, int x, auto& ff_ref) -> void {
        if (y < 0 || y >= h || x < 0 || x >= w || visited[y][x]) return;
        if (img[y][x] > 0.5) return; // white pixel
        visited[y][x] = true;
        ff_ref(y-1, x, ff_ref);
        ff_ref(y+1, x, ff_ref);
        ff_ref(y, x-1, ff_ref);
        ff_ref(y, x+1, ff_ref);
    };
    
    // Mark background as visited (flood fill from edges)
    for (int i = 0; i < h; i++) {
        if (img[i][0] < 0.5 && !visited[i][0]) floodFill(i, 0, floodFill);
        if (img[i][w-1] < 0.5 && !visited[i][w-1]) floodFill(i, w-1, floodFill);
    }
    for (int j = 0; j < w; j++) {
        if (img[0][j] < 0.5 && !visited[0][j]) floodFill(0, j, floodFill);
        if (img[h-1][j] < 0.5 && !visited[h-1][j]) floodFill(h-1, j, floodFill);
    }
    
    // Count remaining black regions (holes)
    int holes = 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (img[i][j] < 0.5 && !visited[i][j]) {
                holes++;
                floodFill(i, j, floodFill);
            }
        }
    }
    return holes;
}

// Calculate features from image
struct Features {
    double top_density;
    double bottom_density;
    double left_density;
    double right_density;
    double center_density;
    double top_left_density;
    double top_right_density;
    double bottom_left_density;
    double bottom_right_density;
    int holes;
    double vertical_symmetry;
    double horizontal_symmetry;
    double aspect_ratio;
    double top_third_density;
    double middle_third_density;
    double bottom_third_density;
};

Features extractFeatures(const IMAGE_T &img) {
    Features f;
    int h = img.size();
    int w = img[0].size();
    
    // Calculate densities in different regions
    auto calcDensity = [&](int y1, int y2, int x1, int x2) -> double {
        double sum = 0;
        int count = 0;
        for (int i = y1; i < y2; i++) {
            for (int j = x1; j < x2; j++) {
                sum += img[i][j];
                count++;
            }
        }
        return count > 0 ? sum / count : 0;
    };
    
    int mid_h = h / 2;
    int mid_w = w / 2;
    int third_h = h / 3;
    
    f.top_density = calcDensity(0, mid_h, 0, w);
    f.bottom_density = calcDensity(mid_h, h, 0, w);
    f.left_density = calcDensity(0, h, 0, mid_w);
    f.right_density = calcDensity(0, h, mid_w, w);
    f.center_density = calcDensity(h/4, 3*h/4, w/4, 3*w/4);
    
    f.top_left_density = calcDensity(0, mid_h, 0, mid_w);
    f.top_right_density = calcDensity(0, mid_h, mid_w, w);
    f.bottom_left_density = calcDensity(mid_h, h, 0, mid_w);
    f.bottom_right_density = calcDensity(mid_h, h, mid_w, w);
    
    f.top_third_density = calcDensity(0, third_h, 0, w);
    f.middle_third_density = calcDensity(third_h, 2*third_h, 0, w);
    f.bottom_third_density = calcDensity(2*third_h, h, 0, w);
    
    // Count holes
    f.holes = countHoles(img);
    
    // Calculate symmetry
    double vert_diff = 0, horiz_diff = 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w/2; j++) {
            vert_diff += std::abs(img[i][j] - img[i][w-1-j]);
        }
    }
    for (int i = 0; i < h/2; i++) {
        for (int j = 0; j < w; j++) {
            horiz_diff += std::abs(img[i][j] - img[h-1-i][j]);
        }
    }
    f.vertical_symmetry = 1.0 - (vert_diff / (h * w / 2));
    f.horizontal_symmetry = 1.0 - (horiz_diff / (h * w / 2));
    
    // Calculate aspect ratio (width of digit)
    int left_bound = w, right_bound = 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (img[i][j] > 0.5) {
                left_bound = std::min(left_bound, j);
                right_bound = std::max(right_bound, j);
            }
        }
    }
    f.aspect_ratio = (right_bound > left_bound) ? (double)(right_bound - left_bound) / w : 0.5;
    
    return f;
}

int judge(IMAGE_T &img) {
    Features f = extractFeatures(img);
    
    // Rule-based classification using extracted features
    
    // 0: Has one hole, good vertical symmetry, balanced top/bottom
    if (f.holes == 1 && f.vertical_symmetry > 0.7 && 
        std::abs(f.top_density - f.bottom_density) < 0.15) {
        return 0;
    }
    
    // 8: Has two holes
    if (f.holes >= 2) {
        return 8;
    }
    
    // 1: Very narrow, high aspect ratio difference, mostly in center
    if (f.aspect_ratio < 0.4 && f.center_density > 0.5) {
        return 1;
    }
    
    // 6: Has one hole, bottom heavy
    if (f.holes == 1 && f.bottom_density > f.top_density + 0.1) {
        return 6;
    }
    
    // 9: Has one hole, top heavy
    if (f.holes == 1 && f.top_density > f.bottom_density + 0.1) {
        return 9;
    }
    
    // 7: Top heavy, strong top third
    if (f.top_third_density > 0.6 && f.top_third_density > f.bottom_third_density + 0.2) {
        return 7;
    }
    
    // 4: Middle and top heavy, has a gap in bottom left
    if (f.bottom_left_density < 0.3 && f.middle_third_density > 0.5 && 
        f.right_density > f.left_density) {
        return 4;
    }
    
    // 2: Bottom heavy, curves
    if (f.bottom_density > f.top_density + 0.05 && f.holes == 0 && 
        f.bottom_left_density > 0.4) {
        return 2;
    }
    
    // 3: Right heavy, no holes, relatively symmetric vertically
    if (f.holes == 0 && f.right_density > f.left_density + 0.1 && 
        std::abs(f.top_density - f.bottom_density) < 0.2) {
        return 3;
    }
    
    // 5: Top heavy, no holes, left-right asymmetric
    if (f.holes == 0 && f.top_density > f.bottom_density && 
        std::abs(f.left_density - f.right_density) > 0.1) {
        return 5;
    }
    
    // Default fallback based on density patterns
    if (f.holes == 1) return 0;
    if (f.holes >= 2) return 8;
    if (f.top_density > f.bottom_density + 0.1) return 7;
    if (f.bottom_density > f.top_density + 0.1) return 2;
    if (f.aspect_ratio < 0.4) return 1;
    
    // Final fallback
    return 3;
}

#endif // SRC_HPP
