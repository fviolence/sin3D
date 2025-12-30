#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <pthread.h>
#include <stdatomic.h>

#include "raylib.h"

#define HEIGHT 720
#define WIDTH 1280

#define NTHREADS 32
#define TILE 32

#define PHASE_SCALE 64*PI*PI
#define Z_CUTOFF 1e-3f
#define X_MAX 1.0f
#define Y_MAX 1.0f
#define X_MIN -X_MAX
#define Y_MIN -Y_MAX
#define X_AXIS_LEN (X_MAX-X_MIN)
#define Y_AXIS_LEN (Y_MAX-Y_MIN)
#define DELTA_MIN 0.015f
#define DELTA_MAX 0.075f

typedef struct Line2D {
    Vector2 startPos;
    Vector2 endPos;
} Line2D;

typedef struct Line3D {
    Vector3 startPos;
    Vector3 endPos;
} Line3D;

typedef struct Triangle3 {
    Vector3 a;
    Vector3 b;
    Vector3 c;
} Triangle3;

typedef struct Vertices3 {
    Triangle3 *items;
    size_t count;
    size_t capacity;
} Vertices3;

typedef struct Triangle2 {
    Vector2 a;
    float invZa;
    Vector2 b;
    float invZb;
    Vector2 c;
    float invZc;
    float depth;  // camera-space depth to sort from far-to-near
    int id;
    int minx, miny, maxx, maxy; // clamped bbox for binning
    Color color;
} Triangle2;

typedef struct Vertices2 {
    Triangle2 *items;
    size_t count;
    size_t capacity;
} Vertices2;

typedef struct ThreadCtx {
    size_t id;
    Vertices2 *vertices2D;
} ThreadCtx;

typedef struct Job {
    // time-varying
    float plot_phase;
    Vector3 camera;
    // plane basis
    Vector3 i, j, k;
    // plane vertices
    Vertices3 *vertices3D;
} Job;

static pthread_t g_threads[NTHREADS];
static ThreadCtx g_ctx[NTHREADS];
static Job g_job; // context for generation job
static Vertices2 *g_vertices2D = NULL; // global pointer to 2D vertices

typedef struct ZBuffer {
    float invZ;
    int id;
} ZBuffer;

// Array of depthes and IDs of triangles for each pixel
static ZBuffer *g_zbuf = NULL;
// Compressed Sparse Row style bin storage
static int *g_tileOffsets = NULL;
static int *g_tileTris = NULL;

static const size_t g_tilesX = (WIDTH  + TILE - 1) / TILE;
static const size_t g_tilesY = (HEIGHT + TILE - 1) / TILE;
static const size_t g_numTiles = g_tilesX * g_tilesY;

static pthread_barrier_t g_barrier_start;
static pthread_barrier_t g_barrier_end;
static atomic_int g_shutdown = 0;
 
typedef enum { STAGE_GEN = 0, STAGE_ZPASS = 1 } Stage;
static atomic_int g_stage = STAGE_GEN;

static inline Vector2 translate2Screen(Vector2 p) {
    // Translate to screen coordinate system: X is right, Y is down, (0,0) is top left corner
    return (Vector2){
        WIDTH  * ( p.x + X_AXIS_LEN / 2) / X_AXIS_LEN,
        HEIGHT * (-p.y + Y_AXIS_LEN / 2) / Y_AXIS_LEN
    };
}

static Vector2 projectOn2DPlane(Vector3 p) {
    // Z coordinates (depth) is projected on XY plane
    if (p.z < Z_CUTOFF) {
        // simple near-plane guard
        int sign_x = ((p.x < 0) ^ (p.z < 0)) ? -1 : 1;
        int sign_y = ((p.y < 0) ^ (p.z < 0)) ? -1 : 1;
        return (Vector2){sign_x * __FLT_MAX__, sign_y * __FLT_MAX__};
    }
    return (Vector2){p.x / p.z, p.y / p.z};
}

static inline Vector3 v3Sub(Vector3 a, Vector3 b) {
    return (Vector3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline float v3Dot(Vector3 a, Vector3 b) {
    // |a|*|b|*cos(alpha)
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vector3 v3Cross(Vector3 a, Vector3 b) {
    // Effectively a vector perpendicular to AB plane
    return (Vector3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static Vector3 v3Norm(Vector3 v) {
    // Calculating on double precision to check for overflow
    double len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (fabs(len) > (double)__FLT_MAX__) return (Vector3){0, 0, 0};
    float inv = 1.0f / len;
    return (Vector3){v.x * inv, v.y * inv, v.z * inv};
}

static inline Vector3 worldToBasis(Vector3 p, Vector3 i, Vector3 j, Vector3 k) {
    return (Vector3) {
        v3Dot(p, i),
        v3Dot(p, j),
        v3Dot(p, k)
    };
}

static inline Vector3 basisToWorld(Vector3 p, Vector3 i, Vector3 j, Vector3 k) {
    // linear combination
    return (Vector3) {
        i.x * p.x + j.x * p.y + k.x * p.z,
        i.y * p.x + j.y * p.y + k.y * p.z,
        i.z * p.x + j.z * p.y + k.z * p.z,
    };
}

static inline float v3Len(Vector3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static void buildOrthonormalBasisFromZ(Vector3 z_in, Vector3 *x_out, Vector3 *y_out, Vector3 *z_out) {
    Vector3 z = v3Norm(z_in);

    if (z.x == 0 && z.y == 0 && z.z == 0) {
        // Degenerate input; return identity basis
        *x_out = (Vector3){1, 0, 0};
        *y_out = (Vector3){0, 1, 0};
        *z_out = (Vector3){0, 0, 1};
        return;
    }

    // Default world up axis is Z
    Vector3 x = v3Norm(v3Cross(z, (Vector3){0, 0, 1}));
    if (x.x == 0 && x.y == 0 && x.z == 0) {
        // world Z is tageting the camera, hence not visible
        // Swtiching to use world Y as worldUp
        x = v3Norm(v3Cross(z, (Vector3){0, 1, 0}));
    }

    // Both 'x' and 'z' are normalized, hence no normalization needed for `y`
    Vector3 y = v3Cross(x, z);

    *x_out = x;
    *y_out = y;
    *z_out = z;
}

static Vector3 cameraViewLookAtOrigin(Vector3 p, const Vector3 camera) {
    // Translation: basically a vector from camera point to the given point in 3D space
    Vector3 d = v3Sub(p, camera);

    // Camera always try to target world origin
    Vector3 target  = (Vector3){0};

    // Constracting basis
    Vector3 x = {0};
    Vector3 y = {0};
    Vector3 z = {0};

    // In world view we use X as right, Y as forward and Z as up
    // But camera's look direction is depth and for further projection
    // on 2D plane camera's coordinate system:
    // * X as right
    // * Y as up
    // * Z as forward
    buildOrthonormalBasisFromZ(v3Sub(target, camera), &x, &z, &y);
    // Convert world coordinates to camera-basis
    return worldToBasis(d, x, z, y);
}

static inline Vector2 translate3DToScreen(Vector3 p, const Vector3 camera) {
    return translate2Screen(projectOn2DPlane(cameraViewLookAtOrigin(p, camera)));
}

static double get_secs() {
    struct timespec tp = {0};
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (double)tp.tv_sec + (double)tp.tv_nsec*1e-9;
}

static void vertices3_append(Vertices3 *to, Triangle3 item) {
    if (to->count == to->capacity) {
        if (to->capacity == 0) to->capacity = 256;
        else to->capacity *= 2;
        to->items = realloc(to->items, sizeof(Triangle3)*to->capacity);
    }
    to->items[to->count++] = item;
}

// Triangulate an XY plane of a given coordinate system
static void triangulateXYPlane(Vertices3 *vertices3D, float delta) {
    vertices3D->count = 0;
    for (float x = -1; x < 1; x += delta)
        for (float y = -1; y < 1; y += delta) {
            // A square: (x, y) -> (x + dt, y) -> (x + dt, y + dt) -> (x, y + dt)
            // this square gives us 2 triangels to add to vertices list
            Vector3 a = {x,         y,      0};
            Vector3 b = {x + delta, y,      0};
            Vector3 c = {x + delta, y + delta, 0};
            Vector3 d = {x,         y + delta, 0};
            // counter-clockwise order !
            vertices3_append(vertices3D, (Triangle3){a, d, c});
            vertices3_append(vertices3D, (Triangle3){c, b, a});
        }
}

static void vertices2_append(Vertices2 *to, Triangle2 item) {
    if (to->count == to->capacity) {
        if (to->capacity == 0) to->capacity = 256;
        else to->capacity *= 2;
        to->items = realloc(to->items, sizeof(Triangle2)*to->capacity);
    }
    to->items[to->count++] = item;
}

static int cmp_triangle2_far_to_near(const void *pa, const void *pb) {
    const Triangle2 *a = (const Triangle2 *)pa;
    const Triangle2 *b = (const Triangle2 *)pb;
    // far-to-near: draw far first, near last
    // returns -1, 0 or 1
    return (a->depth < b->depth) - (a->depth > b->depth);
}

static inline int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline float clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static void addVertice(Vertices2 *to, Vector3 ac, Vector3 bc, Vector3 cc) {
    // ac, bc and cc - vectors in camera's coordinate space
    Vector2 a2 = translate2Screen(projectOn2DPlane(ac));
    Vector2 b2 = translate2Screen(projectOn2DPlane(bc));
    Vector2 c2 = translate2Screen(projectOn2DPlane(cc));

    // Skip vertices outside the screen
    float minx_f = fminf(a2.x, fminf(b2.x, c2.x));
    float maxx_f = fmaxf(a2.x, fmaxf(b2.x, c2.x));
    float miny_f = fminf(a2.y, fminf(b2.y, c2.y));
    float maxy_f = fmaxf(a2.y, fmaxf(b2.y, c2.y));
    if (maxx_f < 0 || maxy_f < 0 || minx_f >= WIDTH || miny_f >= HEIGHT) return;

    // int bbox is half-open: [minx, maxx), [miny, maxy)
    int minx = clampi((int)floorf(minx_f), 0, WIDTH);
    int maxx = clampi((int)ceilf (maxx_f), 0, WIDTH);
    int miny = clampi((int)floorf(miny_f), 0, HEIGHT);
    int maxy = clampi((int)ceilf (maxy_f), 0, HEIGHT);
    if (minx >= maxx || miny >= maxy) return;

    float cross = (b2.x - a2.x) * (c2.y - a2.y) - (b2.y - a2.y) * (c2.x - a2.x);
    // Collinear vertices - skip for now
    if (cross == 0.0f) return;

    Triangle2 item = {0};
    if (cross < 0) {
        item.a = a2; item.b = b2; item.c = c2;
        item.invZa = 1.0f / ac.z;
        item.invZb = 1.0f / bc.z;
        item.invZc = 1.0f / cc.z;
        item.color = PINK;
    } else {
        // swap b/c to keep consistent winding
        item.a = a2; item.b = c2; item.c = b2;
        item.invZa = 1.0f / ac.z;
        item.invZb = 1.0f / cc.z;
        item.invZc = 1.0f / bc.z;
        item.color = GREEN;
    }

    item.depth = ac.z + bc.z + cc.z;  // good enough approximation for depth for painter sorting key

    item.minx = minx;
    item.maxx = maxx;

    item.miny = miny;
    item.maxy = maxy;

    item.id = 0;

    vertices2_append(to, item);
}

static inline Vector3 lerp3(Vector3 a, Vector3 b, float t) {
    return (Vector3){
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    };
}

static Vector3 intersectWithNearZ(Vector3 p, Vector3 q) {
    // signed Z-distance
    float denom = (q.z - p.z);
    // denom should not be 0 if p and q are on different sides, but guard anyway.
    if (fabsf(denom) < 1e-12f) return p;
    // signed Z-distance proportion to cutoff plane
    float t = (Z_CUTOFF - p.z) / denom;
    // Optional clamp for numerical safety:
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    // r(t) = p + t * (q − p), t ∈ [0,1]
    return lerp3(p, q, t);
}

static size_t clipPolyNearPlane(Vector3 *points_in, Vector3 *points_out) {
    // Sutherland–Hodgman clipping
    size_t out_cnt = 0;

    for (size_t cnt = 0; cnt < 3; cnt++) {
        Vector3 curr = points_in[cnt];
        Vector3 next = points_in[(cnt + 1) % 3];

        int currIn = (curr.z >= Z_CUTOFF);
        int nextIn = (next.z >= Z_CUTOFF);

        if (currIn && nextIn) {
            // in -> in : keep next
            points_out[out_cnt++] = next;
        } else if (currIn && !nextIn) {
            // in -> out : keep intersection
            points_out[out_cnt++] = intersectWithNearZ(curr, next);
        } else if (!currIn && nextIn) {
            // out -> in : keep intersection and next
            points_out[out_cnt++] = intersectWithNearZ(curr, next);
            points_out[out_cnt++] = next;
        } else {
            // out -> out : keep nothing
        }
    }

    return out_cnt;
}

// Edge function: Z component of a cross product of vectors AC and AB in 3D space with z=0
// Result is a signed float value equal, which absolute value is an area of ​​a parallelogram with sides AB and AC
// And sing is a direction of bypass, which is defined by order of arguments and by direction of Y coordinate
// For screen space with Y growing down and (AC x AB): negative - CCW, positive - CW
static inline float edgef(float ax, float ay, float bx, float by, float cx, float cy) {
    return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
}

static void raster_tri_depth_tile(const Triangle2 *t,
                                  int x0, int y0, int x1, int y1,
                                  ZBuffer *zbuf, int W)
{
    // For each pixel in the tile rectangle, determine whether the pixel lies inside triangle t.
    // If yes, compute an interpolated depth measure and update zbuf
    float ax = t->a.x, ay = t->a.y;
    float bx = t->b.x, by = t->b.y;
    float cx = t->c.x, cy = t->c.y;

    float area = edgef(ax, ay, bx, by, cx, cy);
    if (area == 0.0f) return;

    int areaPos = (area > 0.0f);
    float invArea = 1.0f / area;

    // Clip tile bounds to triangle bbox to avoid extra work
    if (x0 < t->minx) x0 = t->minx;
    if (y0 < t->miny) y0 = t->miny;
    if (x1 > t->maxx) x1 = t->maxx;
    if (y1 > t->maxy) y1 = t->maxy;
    if (x0 >= x1 || y0 >= y1) return;

    // Check each pixel inside the clipped box if it's inside the triangle
    for (int y = y0; y < y1; y++) {
        // The pixel is filled in the [x, x + 1] and [y, y + 1] positions by X and Y, respectively.
        // Operating on the centre point of the pixel, hence +0.5
        float py = (float)y + 0.5f;
        int row = y * W;

        for (int x = x0; x < x1; x++) {
            float px = (float)x + 0.5f;

            // Determine on which side of the oriented edge the point lies
            float w0 = edgef(bx, by, cx, cy, px, py);
            float w1 = edgef(cx, cy, ax, ay, px, py);
            float w2 = edgef(ax, ay, bx, by, px, py);

            /* For a point to be inside a figure, all the small triangles formed
             * by the point and each edge of the figure must be in the same
             * bypass direction and coincide with the bypass direction of the
             * figure itself. So a point is inside if w0,w1,w2 have all the same
             * sign. Additionally this sign will match a sign of an area.
             */
            if (areaPos) { if (w0 < 0 || w1 < 0 || w2 < 0) continue; }
            else         { if (w0 > 0 || w1 > 0 || w2 > 0) continue; }

            // Normalized weights
            w0 *= invArea;
            w1 *= invArea;
            w2 *= invArea;

            // Calculate weighted invZ, test and update zbuf
            float invZ = w0 * t->invZa + w1 * t->invZb + w2 * t->invZc;
            int idx = row + x;
            if (invZ > zbuf[idx].invZ) {
                zbuf[idx].invZ = invZ;
                zbuf[idx].id = t->id;
            }
        }
    }
}

static void build_tile_bins(const Vertices2 *vertices2D) {
    // Effectively matrix [g_tilesX x g_tilesY]
    int *counts = calloc(g_numTiles, sizeof(int));
    for (size_t i = 0; i < vertices2D->count; ++i) {
        const Triangle2 *t = &vertices2D->items[i];

        // Convert bbox into a tile range
        int tx0 =  t->minx / TILE;
        int ty0 =  t->miny / TILE;
        int tx1 = (t->maxx - 1) / TILE;
        int ty1 = (t->maxy - 1) / TILE;

        // Count how many triangles go into each tile
        for (int ty = ty0; ty <= ty1; ty++)
            for (int tx = tx0; tx <= tx1; tx++)
                counts[ty * (int)g_tilesX + tx]++;
    }

    // a. tileOffsets[t] number of triangles in the tileTris array prio to the tile t
    // b. (tileOffsets[t + 1] - tileOffsets[t]) number of triangles in the tile t,
    //    that is size of the bin is the tileTris for a given tile t
    // c. last tileOffsets entry is a number of all triangles that fit in all tiles
    g_tileOffsets[0] = 0;
    for (size_t t = 0; t < g_numTiles; t++)
        g_tileOffsets[t + 1] = g_tileOffsets[t] + counts[t];
    // Flat array split into bins containing triangle indices crossing each tile, packed tile-by-tile
    g_tileTris = realloc(g_tileTris, sizeof(int) * g_tileOffsets[g_numTiles]);
    free(counts);

    // Cursor indicating next free spot in tileTris for each bin
    int *write = malloc(sizeof(int) * g_numTiles);
    memcpy(write, g_tileOffsets, sizeof(int) * g_numTiles);

    // Fill triangle indecies array
    for (size_t i = 0; i < vertices2D->count; i++) {
        const Triangle2 *t = &vertices2D->items[i];

        int tx0 =  t->minx / TILE;
        int ty0 =  t->miny / TILE;
        int tx1 = (t->maxx - 1) / TILE;
        int ty1 = (t->maxy - 1) / TILE;

        for (int ty = ty0; ty <= ty1; ty++)
            for (int tx = tx0; tx <= tx1; tx++) {
                int tile = ty * (int)g_tilesX + tx;
                // progress the cursor of the bin
                g_tileTris[ write[tile]++ ] = (int)i;
            }
    }

    free(write);
}

// default sinus
static inline float plot_function_sin(float x, float y, float phase) {
    return sinf(sqrtf(PHASE_SCALE * (x * x + y * y)) - phase) / 3;
}

// water drop
static inline float plot_function_dropplet(float x, float y, float phase) {
    float r = sqrtf(x*x + y*y);
    return 0.35f * sinf(18.0f * r - phase) / (1.0f + 2.0f * r);
}

// Gaussian-modulated wave packet
static inline float plot_function_Gauss(float x, float y, float phase) {
    float r2 = x * x + y * y;
    return 0.5f * expf(-3.0f * r2) * sinf(25.0f * sqrtf(r2) - phase);
}

// Mexican hat
static inline float plot_function_mexican(float x, float y, float phase) {
    float r2 = x * x + y * y;
    float a = 4.0f + 1.5f * sinf(0.7f * phase);
    return 0.8f * (1.0f - a * r2) * expf(-a * r2);
}

// Interference pattern
static inline float plot_function_moire(float x, float y, float phase) {
    float x1 = 0.4f, y1 = 0.0f;
    float x2 = -0.4f, y2 = 0.0f;
    float r1 = sqrtf((x - x1) * (x - x1) + (y - y1) * (y - y1));
    float r2 = sqrtf((x - x2) * (x - x2) + (y - y2) * (y - y2));
    return 0.18f * (sinf(22.0f * r1 - phase) + sinf(22.0f * r2 - phase));
}

// spiky star
static inline float plot_function_star(float x, float y, float phase) {
    float r2 = x * x + y * y;
    float th = atan2f(y, x);
    return 0.45f * sinf(6.0f * th + 0.8f * phase) * expf(-2.5f * r2);
}

static void *worker_main(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    size_t id = ctx->id;

    for (;;) {
        // Wait until main publishes a new job
        pthread_barrier_wait(&g_barrier_start);
        if (atomic_load(&g_shutdown)) break;

        int stage = atomic_load(&g_stage);
        if (stage == STAGE_GEN) {
            const Job *job = &g_job;
            Vertices3 *vertices3D = job->vertices3D;
            Vertices2 *vertices2D = ctx->vertices2D;

            size_t total = vertices3D->count;
            size_t chunk = (total + NTHREADS - 1) / NTHREADS;
            size_t start = id * chunk;
            size_t end = start + chunk;
            if (end > total) end = total;

            vertices2D->count = 0;
            Vector3 points_in[3] = {0};
            Vector3 points_out[4] = {0};

            for (size_t n = start; n < end; n++) {
                // Local plane coordinates
                points_in[0] = vertices3D->items[n].a;
                points_in[1] = vertices3D->items[n].b;
                points_in[2] = vertices3D->items[n].c;
                for (size_t cnt = 0; cnt < 3; cnt++) {
                    // apply z(x,y) function
                    Vector3 *v = &points_in[cnt];
                    v->z = plot_function_dropplet(v->x, v->y, job->plot_phase);
                    // world coordinates
                    *v = basisToWorld(*v, job->i, job->j, job->k);
                    // Translate points to camera view space
                    *v = cameraViewLookAtOrigin(*v, job->camera);
                }

                // cut/skip if plane near/behind depth cutoff plane
                switch (clipPolyNearPlane(points_in, points_out)) {
                    case 0:
                        // Simple near-plane cull: full triangle is behind/too close, skip.
                        continue;
                    case 3:
                        addVertice(vertices2D, points_out[0], points_out[1], points_out[2]);
                        break;
                    case 4:
                        addVertice(vertices2D, points_out[0], points_out[1], points_out[2]);
                        addVertice(vertices2D, points_out[2], points_out[3], points_out[0]);
                        break;
                    default:
                        printf("UNREACHABLE in plotter in worker: %zu\n", id);
                        abort();
                }
            }
        } else { // STAGE_ZPASS
            // Each worker owns a disjoint set of tiles
            size_t tilesPerThread = (g_numTiles + NTHREADS - 1) / NTHREADS;
            size_t tileStart = id * tilesPerThread;
            size_t tileEnd = tileStart + tilesPerThread;
            if (tileEnd > g_numTiles) tileEnd = g_numTiles;
            // each thread process some amount of tiles
            for (size_t tile = tileStart; tile < tileEnd; tile++) {
                int tx = (int)(tile % g_tilesX);
                int ty = (int)(tile / g_tilesX);

                int x0 = tx * TILE;
                int y0 = ty * TILE;

                int x1 = x0 + TILE;
                if (x1 > WIDTH) x1 = WIDTH;

                int y1 = y0 + TILE;
                if (y1 > HEIGHT) y1 = HEIGHT;

                // tile related bin triangle indecies
                int begin = g_tileOffsets[tile];
                int end   = g_tileOffsets[tile + 1];

                // raster each triangle in a bin
                for (int k = begin; k < end; k++) {
                    int triIndex = g_tileTris[k];
                    const Triangle2 *t = &g_vertices2D->items[triIndex];
                    raster_tri_depth_tile(t, x0, y0, x1, y1, g_zbuf, WIDTH);
                }
            }
        }

        // Signal done
        pthread_barrier_wait(&g_barrier_end);
    }

    // Let main pass the end barrier too if it uses the same lifecycle
    pthread_barrier_wait(&g_barrier_end);
    return NULL;
}

static void shutdown_workers() {
    atomic_store(&g_shutdown, 1);

    // Release workers so they can exit
    pthread_barrier_wait(&g_barrier_start);
    pthread_barrier_wait(&g_barrier_end);

    for (int i = 0; i < NTHREADS; ++i) pthread_join(g_threads[i], NULL);
    pthread_barrier_destroy(&g_barrier_start);
    pthread_barrier_destroy(&g_barrier_end);

    for (int i = 0; i < NTHREADS; ++i) {
        free(g_ctx[i].vertices2D->items);
        free(g_ctx[i].vertices2D);
    }
}

static void init_workers() {
    pthread_barrier_init(&g_barrier_start, NULL, NTHREADS + 1);
    pthread_barrier_init(&g_barrier_end, NULL, NTHREADS + 1);
    for (size_t id = 0; id < NTHREADS; id++) {
        g_ctx[id].id = id;
        g_ctx[id].vertices2D = calloc(1, sizeof(Vertices2));
        pthread_create(&g_threads[id], NULL, worker_main, &g_ctx[id]);
    }
}

static void plot3D(Vertices2 *vertices2D, Vertices3 *vertices3D, const Vector3 camera, Vector3 i, Vector3 j, Vector3 k, float plot_phase) {
    // Stage 0: Prepare 2D vertices plane
    // Camera in basis coords (u,v,w)
    Vector3 uvw = worldToBasis(camera, i, j, k);
    // Clamp u,v to finite patch bounds
    float uc = clampf(uvw.x, X_MIN, X_MAX);
    float vc = clampf(uvw.y, Y_MIN, Y_MAX);
    // Closest point on the patch in world coords (on plane => w=0)
    Vector3 closest = basisToWorld((Vector3){uc, vc, 0.0f}, i, j, k);
    // Euclidean distance to that point
    float dist = v3Len(v3Sub(camera, closest));
    // Delta is define by the distance from the view point to the finite plane patch
    triangulateXYPlane(vertices3D, clampf(dist / 100, DELTA_MIN, DELTA_MAX));

    // Stage 1: generate triangles
    g_job = (Job){
        .plot_phase = plot_phase,
        .i = i,
        .j = j,
        .k = k,
        .camera = camera,
        .vertices3D = vertices3D
    };

    // Release workers for Gen stage and wait completion
    atomic_store(&g_stage, STAGE_GEN);
    pthread_barrier_wait(&g_barrier_start);
    pthread_barrier_wait(&g_barrier_end);

    // merge
    vertices2D->count = 0;
    for (int i = 0; i < NTHREADS; ++i)
        for (size_t n = 0; n < g_ctx[i].vertices2D->count; n++)
            vertices2_append(vertices2D, g_ctx[i].vertices2D->items[n]);

    // Assign stable ids for Z-pass and later visibility
    for (size_t n = 0; n < vertices2D->count; n++) vertices2D->items[n].id = (int)n;

    // Stage 2: Z-pass raster (occlusion mask)
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        g_zbuf[i].invZ = -__FLT_MAX__;
        g_zbuf[i].id = -1;
    }
    build_tile_bins(vertices2D);

    // Release workers for zPass stage and wait completion
    g_vertices2D = vertices2D;
    atomic_store(&g_stage, STAGE_ZPASS);
    pthread_barrier_wait(&g_barrier_start);
    pthread_barrier_wait(&g_barrier_end);

    // Build visibility mask
    unsigned char *visible = calloc(vertices2D->count, sizeof(char));
    for (int p = 0; p < WIDTH * HEIGHT; p++) {
        int id = g_zbuf[p].id;
        if (id >= 0) visible[id] = 1;
    }

    // And filter out non-visible
    size_t out = 0;
    for (size_t n = 0; n < vertices2D->count; n++)
        if (visible[vertices2D->items[n].id]) vertices2D->items[out++] = vertices2D->items[n];
    vertices2D->count = out;
    free(visible);

    // Stage 3: painter sort + draw
    qsort(vertices2D->items, vertices2D->count, sizeof(Triangle2), cmp_triangle2_far_to_near);

    // Draw the vertices in sorted order, with the farthest ones drawn first.
    // This ensures that the closest ones are always on top of all the others.
    // This avoids the image being flipped upside down every 180 degrees of camera rotation.
    for (size_t n = 0; n < vertices2D->count; n++) {
        DrawTriangle(vertices2D->items[n].a, vertices2D->items[n].b, vertices2D->items[n].c, vertices2D->items[n].color);
        DrawTriangleLines(vertices2D->items[n].a, vertices2D->items[n].b, vertices2D->items[n].c, BLACK);
    }
}

static Vector3 orbit_pitch_around_origin(Vector3 camera, float r, float vPitch, float hPitch) {
    float x = camera.x, y = camera.y, z = camera.z;

    // horizontal angle around Z
    float yaw   = atan2f(y, x);
    // distance of camera from Z axis
    float horiz = sqrtf(x*x + y*y);
    // elevation angle above the horizontal plane
    float pitch = atan2f(z, horiz);

    // adjust vertical pitch, while avoiding flipping at the poles
    pitch += vPitch;
    const float eps = 1e-3f;
    pitch = clampf(pitch, -(PI / 2 - eps), PI / 2 - eps);

    // horizontal rotation
    yaw += hPitch;

    float ch = cosf(pitch);
    return (Vector3){
        r * ch * cosf(yaw),
        r * ch * sinf(yaw),
        r * sinf(pitch)
    };
}

int main(void) {
    InitWindow(WIDTH, HEIGHT, "Plot 3D");
    SetTargetFPS(60);

    g_zbuf = calloc(WIDTH * HEIGHT, sizeof(ZBuffer));
    g_tileOffsets = calloc(g_numTiles + 1, sizeof(int));

    Vector3 camera = {0.5, -0.5, 1.5};
    float camera_radius = v3Len(camera);

    Vertices2 vertices2D = {0}; // Set of vertices in screen space
    Vertices3 vertices3D = {0}; // Tringulated plane in 3D space passed to a plot function

    Line3D xAxis = { .startPos = (Vector3){-1, 0, 0}, .endPos = (Vector3){1, 0, 0} };
    Line3D yAxis = { .startPos = (Vector3){0, -1, 0}, .endPos = (Vector3){0, 1, 0} };
    Line3D zAxis = { .startPos = (Vector3){0, 0, -1}, .endPos = (Vector3){0, 0, 1} };

    Line3D xRightBorder = { .startPos = (Vector3){-1, 1, 0}, .endPos = (Vector3){1, 1, 0} };
    Line3D xLeftBorder = { .startPos = (Vector3){-1, -1, 0}, .endPos = (Vector3){1, -1, 0} };

    Line3D yRightBorder = { .startPos = (Vector3){1, -1, 0}, .endPos = (Vector3){1, 1, 0} };
    Line3D yLeftBorder = { .startPos = (Vector3){-1, -1, 0}, .endPos = (Vector3){-1, 1, 0} };

    Vector3 right = {0};
    Vector3 forward = {0};
    Vector3 up = v3Norm((Vector3){1, 0.5, 1.5});
    buildOrthonormalBasisFromZ(up, &right, &forward, &up);

    init_workers();

    float cam_angle = 0.01f;
    float plot_phase = 0;
    size_t itter_count = 0;

    double start_time = get_secs();
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(GRAY);

        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            camera_radius *= powf(0.9f, wheel);   // smooth exponential zoom
            camera_radius = clampf(camera_radius, 0.5f, 15.0f);
        }

        int turn_up = IsKeyDown(KEY_UP) ? 1 : 0;
        int turn_down = IsKeyDown(KEY_DOWN) ? 1 : 0;
        int turn_camera_vert = turn_up - turn_down;

        int turn_left = IsKeyDown(KEY_LEFT) ? 1 : 0;
        int turn_right = IsKeyDown(KEY_RIGHT) ? 1 : 0;
        int turn_camera_hor = turn_right - turn_left;

        camera = orbit_pitch_around_origin(camera, camera_radius, cam_angle * turn_camera_vert, cam_angle * turn_camera_hor);

        plot3D(&vertices2D, &vertices3D, camera, forward, right, up, plot_phase);
        plot_phase += 0.1;

        DrawLineEx(translate3DToScreen((Vector3){0, 0, 0}, camera), translate3DToScreen(xAxis.endPos, camera), 2.0, RED);
        DrawLineEx(translate3DToScreen((Vector3){0, 0, 0}, camera), translate3DToScreen(yAxis.endPos, camera), 2.0, YELLOW);
        DrawLineEx(translate3DToScreen((Vector3){0, 0, 0}, camera), translate3DToScreen(zAxis.endPos, camera), 2.0, BLUE);

        DrawLineEx(translate3DToScreen(xAxis.startPos, camera), translate3DToScreen((Vector3){0, 0, 0}, camera), 2.0, BLACK);
        DrawLineEx(translate3DToScreen(yAxis.startPos, camera), translate3DToScreen((Vector3){0, 0, 0}, camera), 2.0, BLACK);
        DrawLineEx(translate3DToScreen(zAxis.startPos, camera), translate3DToScreen((Vector3){0, 0, 0}, camera), 2.0, BLACK);

        DrawLineEx(translate3DToScreen(xRightBorder.startPos, camera), translate3DToScreen(xRightBorder.endPos, camera), 2.0, BLACK);
        DrawLineEx(translate3DToScreen(xLeftBorder.startPos, camera), translate3DToScreen(xLeftBorder.endPos, camera), 2.0, BLACK);
        DrawLineEx(translate3DToScreen(yRightBorder.startPos, camera), translate3DToScreen(yRightBorder.endPos, camera), 2.0, BLACK);
        DrawLineEx(translate3DToScreen(yLeftBorder.startPos, camera), translate3DToScreen(yLeftBorder.endPos, camera), 2.0, BLACK);

        itter_count++;

        EndDrawing();
    }

    shutdown_workers();

    printf("Avg FPS: %.05lf\n", itter_count / (get_secs() - start_time));

    CloseWindow();

    return 0;
}