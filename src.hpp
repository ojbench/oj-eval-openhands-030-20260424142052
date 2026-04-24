#ifndef SRC_HPP
#define SRC_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>

// Helper function to count connected components (holes)
int countHoles(const std::vector<std::vector<double> > &img) {
    int h = img.size();
    int w = img[0].size();
    std::vector<std::vector<bool> > visited(h, std::vector<bool>(w, false));
    
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

Features extractFeatures(const std::vector<std::vector<double> > &img) {
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

int judge(std::vector<std::vector<double> > &img) {
    Features f = extractFeatures(img);
    
    // More sophisticated rule-based classification
    
    // 1: Very narrow, mostly vertical
    if (f.aspect_ratio < 0.35) {
        return 1;
    }
    
    // 8: Has two holes
    if (f.holes >= 2) {
        return 8;
    }
    
    // 7: Top heavy, strong top, weak bottom
    if (f.top_third_density > 0.55 && f.bottom_third_density < 0.4 && f.holes == 0) {
        return 7;
    }
    
    // 4: Gap in bottom left, right side strong
    if (f.bottom_left_density < 0.35 && f.right_density > 0.5 && f.holes == 0) {
        return 4;
    }
    
    // 5: Top heavy, no holes, right side weaker than left in top half
    if (f.holes == 0 && f.top_density > f.bottom_density + 0.08 && 
        f.top_left_density > f.top_right_density && f.bottom_left_density < f.bottom_right_density) {
        return 5;
    }
    
    // 6: Has one hole, bottom heavy
    if (f.holes == 1 && f.bottom_density > f.top_density + 0.05) {
        return 6;
    }
    
    // 9: Has one hole, top heavy
    if (f.holes == 1 && f.top_density > f.bottom_density + 0.05) {
        return 9;
    }
    
    // 0: Has one hole, balanced
    if (f.holes == 1) {
        return 0;
    }
    
    // 2: Bottom heavy, no holes, left side strong at bottom
    if (f.holes == 0 && f.bottom_density > f.top_density + 0.05 && 
        f.bottom_left_density > 0.45) {
        return 2;
    }
    
    // 3: No holes, relatively balanced but right-leaning
    if (f.holes == 0 && f.right_density > f.left_density) {
        return 3;
    }
    
    // Remaining fallbacks
    if (f.holes == 0 && f.top_density > f.bottom_density + 0.1) return 7;
    if (f.holes == 0 && f.bottom_density > f.top_density + 0.1) return 2;
    
    // Final fallback
    return 3;
}

#endif // SRC_HPP
