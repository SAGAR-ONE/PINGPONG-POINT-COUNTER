import com.github.sarxos.webcam.Webcam;
import javax.swing.*;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Table Tennis Tracker — Complete Edition
 *
 * Features:
 *  1.  Ball registration (30-frame HSV sampling)
 *  2.  Perspective correction via homography
 *  3.  Exponential smoothing on ball position
 *  4.  Trajectory prediction + ghost circle when ball lost
 *  5.  Bounce detection (2 bounces same side = point)
 *  6.  Net crossing + rally counter
 *  7.  Net let detection (quick double-crossing)
 *  8.  Deuce & advantage rules (win by 2 from 10-10)
 *  9.  Best-of-N sets (3 / 5 / 7 / Single)
 *  10. Serve rotation every 2 points
 *  11. 3-2-1 countdown before each set
 *  12. Pause / Resume  (P key)
 *  13. Re-register ball mid-game  (B key)
 *  14. Undo last point  (Z key)
 *  15. Manual score adjust  ([ / ] = P1 ±1,  - / = = P2 ±1)
 *  16. Ball speed display (km/h)
 *  17. Full stats screen on game over
 *  18. Match log exported to match_log.csv
 */
public class App {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    // ════════════════════════════════════════════════════════════════════
    // ── Inner types ──────────────────────────────────────────────────────

    static class BallResult {
        final Point center; final double radius;
        BallResult(Point c, double r) { center = c; radius = r; }
    }

    static class PointRecord {
        final String scorer; final int p1score, p2score, rallyLength;
        final String timestamp;
        PointRecord(String s, int p1, int p2, int rally) {
            scorer = s; p1score = p1; p2score = p2; rallyLength = rally;
            timestamp = LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        }
    }

    static class MatchStats {
        int totalPointsPlayed = 0, longestRally = 0, currentRally = 0;
        int p1MaxConsecutive  = 0, p2MaxConsecutive = 0;
        int p1Consecutive     = 0, p2Consecutive    = 0;
        long totalRallyFrames = 0; int rallyCount    = 0;

        void recordPoint(String scorer) {
            totalPointsPlayed++;
            if (currentRally > longestRally) longestRally = currentRally;
            totalRallyFrames += currentRally; rallyCount++;
            currentRally = 0;
            if (scorer.equals("P1")) {
                p1Consecutive++; p2Consecutive = 0;
                if (p1Consecutive > p1MaxConsecutive) p1MaxConsecutive = p1Consecutive;
            } else {
                p2Consecutive++; p1Consecutive = 0;
                if (p2Consecutive > p2MaxConsecutive) p2MaxConsecutive = p2Consecutive;
            }
        }

        void undoPoint(String scorer) {
            totalPointsPlayed = Math.max(0, totalPointsPlayed - 1);
            if (scorer.equals("P1")) { p1Consecutive = Math.max(0, p1Consecutive - 1); }
            else                     { p2Consecutive = Math.max(0, p2Consecutive - 1); }
        }

        void netCrossing() { currentRally++; }

        double avgRallyLength() {
            return rallyCount == 0 ? 0 : (double) totalRallyFrames / rallyCount;
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Game state ────────────────────────────────────────────────────────

    enum GameState { CONTROLS_SCREEN, REGISTER_BALL, SET_TABLE, COUNTDOWN, PLAYING, PAUSED, GAME_OVER }
    static volatile GameState state         = GameState.CONTROLS_SCREEN;
    static          GameState stateBeforePause = GameState.PLAYING;

    // ── Controls screen ───────────────────────────────────────────────────
    static volatile boolean anyKeyPressed = false;

    // ── Camera adjustments (applied per-frame in software) ────────────────
    static volatile double camBrightness   =   0.0;  // -100 to +100
    static volatile double camContrast     =   1.0;  //  0.5 to  2.0
    static volatile double camSaturation   =   1.0;  //  0.0 to  2.0
    static volatile double camWarmth       =   0.0;  // -50 (cool) to +50 (warm)

    // ── Players ───────────────────────────────────────────────────────────
    static String player1Name = "Player 1";
    static String player2Name = "Player 2";
    static volatile int scoreP1 = 0, scoreP2 = 0;

    // ── Sets ──────────────────────────────────────────────────────────────
    static int TOTAL_SETS = 3;
    static int setsToWin  = 2;
    static int setsP1     = 0, setsP2 = 0;

    // ── Ball HSV ──────────────────────────────────────────────────────────
    static String ballColor = "White";
    static Scalar ballLower = new Scalar(0,   0, 180);
    static Scalar ballUpper = new Scalar(180, 50, 255);

    // ── Ball registration ─────────────────────────────────────────────────
    static double registeredMinRadius = 0, registeredMaxRadius = 0;
    static final List<Double> ballSamples = new ArrayList<>();
    static final int SAMPLE_COUNT = 30;

    // ── Table ─────────────────────────────────────────────────────────────
    static final List<Point> tableCorners = new ArrayList<>();
    static boolean tableSet = false;

    // ── Homography ────────────────────────────────────────────────────────
    static Mat homographyMatrix = null, homographyInverse = null;
    static final int FLAT_W = 640, FLAT_H = 320;

    // ── Trajectory ────────────────────────────────────────────────────────
    static final int TRAJ_HISTORY = 8;
    static final LinkedList<Point> posHistory = new LinkedList<>();
    static Point predictedPos = null;

    // ── Bounce detection ──────────────────────────────────────────────────
    static double prevFlatY = -1, prevDY = 0;
    static String bounceSide = ""; static int bounceCount = 0;
    static final int BOUNCE_THRESHOLD = 2;

    // ── Net let detection ─────────────────────────────────────────────────
    static boolean justCrossed   = false;
    static int     netCrossFrames = 0;

    // ── Scoring ───────────────────────────────────────────────────────────
    static volatile String lastSide         = "";
    static volatile long   lastPointTime    = 0;
    static volatile String pointMessage     = "";
    static volatile long   pointMessageTime = 0;
    static volatile int    ballMissingFrames = 0;
    static final    int    MISSING_THRESHOLD = 25;
    static boolean wasOnP1Side = false, ballSeenOnce = false;

    // ── Smoothing ─────────────────────────────────────────────────────────
    static Point  smoothedCenter = null;
    static double smoothedRadius = 0;
    static final double ALPHA = 0.4;

    // ── Speed ─────────────────────────────────────────────────────────────
    static double ballSpeedKmh    = 0;
    static final double TABLE_LENGTH_M = 2.74;  // real table length in metres
    static final double FPS            = 30.0;

    // ── Serve ─────────────────────────────────────────────────────────────
    static String servingPlayer  = "P1";
    static int    totalPointsDone = 0;

    // ── Countdown ────────────────────────────────────────────────────────
    static long countdownStart  = 0;
    static final int COUNTDOWN_SECS = 3;

    // ── Stats & log ──────────────────────────────────────────────────────
    static final MatchStats stats     = new MatchStats();
    static final LinkedList<PointRecord> undoStack  = new LinkedList<>();
    static final List<PointRecord>       matchLog   = new ArrayList<>();
    static PrintWriter logWriter = null;

    // ── Key flags ─────────────────────────────────────────────────────────
    static volatile boolean restartRequested      = false;
    static volatile boolean pauseRequested        = false;
    static volatile boolean reregisterRequested   = false;
    static volatile boolean undoRequested         = false;
    static volatile boolean cameraSettingsRequested = false;
    static volatile int     p1ScoreAdjust         = 0;
    static volatile int     p2ScoreAdjust         = 0;

    // ════════════════════════════════════════════════════════════════════
    // ── main ─────────────────────────────────────────────────────────────

    public static void main(String[] args) throws InterruptedException {

        // ── Setup dialog ──────────────────────────────────────────────────
        JTextField name1Field = new JTextField("Player 1");
        JTextField name2Field = new JTextField("Player 2");
        String[] colors     = {"White", "Orange", "Yellow"};
        String[] setOptions = {"Best of 3", "Best of 5", "Best of 7", "Single Set"};
        JComboBox<String> colorBox = new JComboBox<>(colors);
        JComboBox<String> setsBox  = new JComboBox<>(setOptions);

        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.add(new JLabel("Player 1 Name (Left Side):")); panel.add(name1Field);
        panel.add(Box.createVerticalStrut(8));
        panel.add(new JLabel("Player 2 Name (Right Side):")); panel.add(name2Field);
        panel.add(Box.createVerticalStrut(8));
        panel.add(new JLabel("Ball Color:")); panel.add(colorBox);
        panel.add(Box.createVerticalStrut(8));
        panel.add(new JLabel("Match Format:")); panel.add(setsBox);

        int dlg = JOptionPane.showConfirmDialog(null, panel,
            "Table Tennis Tracker - Setup",
            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
        if (dlg != JOptionPane.OK_OPTION) { System.out.println("Cancelled."); return; }

        player1Name = name1Field.getText().trim().isEmpty() ? "Player 1" : name1Field.getText().trim();
        player2Name = name2Field.getText().trim().isEmpty() ? "Player 2" : name2Field.getText().trim();
        ballColor   = (String) colorBox.getSelectedItem();

        switch ((String) setsBox.getSelectedItem()) {
            case "Best of 5":  TOTAL_SETS = 5; break;
            case "Best of 7":  TOTAL_SETS = 7; break;
            case "Single Set": TOTAL_SETS = 1; break;
            default:           TOTAL_SETS = 3;
        }
        setsToWin = TOTAL_SETS / 2 + 1;

        switch (ballColor) {
            case "Orange":
                ballLower = new Scalar(5,  150, 150);
                ballUpper = new Scalar(20, 255, 255); break;
            case "Yellow":
                ballLower = new Scalar(20, 100, 100);
                ballUpper = new Scalar(35, 255, 255); break;
            default:
                ballLower = new Scalar(0,   0,  180);
                ballUpper = new Scalar(180, 50, 255);
        }

        // Open match log
        try {
            logWriter = new PrintWriter(new FileWriter("match_log.csv", true));
            logWriter.println("Timestamp,Scorer,P1Score,P2Score,RallyLength");
            logWriter.flush();
        } catch (IOException ex) {
            System.out.println("Warning: could not open match_log.csv - " + ex.getMessage());
        }

        // ── Camera ────────────────────────────────────────────────────────
        Webcam webcam = Webcam.getDefault();
        webcam.setCustomViewSizes(new Dimension(640, 480));
        webcam.setViewSize(new Dimension(640, 480));
        webcam.open();
        System.out.println("Camera opened.");

        // ── Camera Settings Dialog (live preview) ─────────────────────────
        showCameraSettingsDialog(webcam);

        // ── Window ────────────────────────────────────────────────────────
        JFrame window = new JFrame("Table Tennis Tracker");
        JLabel imageLabel = new JLabel();
        window.add(imageLabel);
        window.setSize(660, 540);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);

        window.addKeyListener(new java.awt.event.KeyAdapter() {
            @Override public void keyPressed(java.awt.event.KeyEvent e) {
                // Any key dismisses the controls screen
                if (state == GameState.CONTROLS_SCREEN) { anyKeyPressed = true; return; }
                char k = Character.toUpperCase(e.getKeyChar());
                switch (k) {
                    case 'C': cameraSettingsRequested = true; break;
                    case 'R': restartRequested    = true; break;
                    case 'P': pauseRequested      = true; break;
                    case 'B': reregisterRequested = true; break;
                    case 'Z': undoRequested       = true; break;
                    case '[': p1ScoreAdjust       = -1;   break;
                    case ']': p1ScoreAdjust       = +1;   break;
                    case '-': p2ScoreAdjust       = -1;   break;
                    case '=': p2ScoreAdjust       = +1;   break;
                }
            }
        });
        window.setFocusable(true);
        window.requestFocus();

        imageLabel.addMouseListener(new MouseAdapter() {
            @Override public void mouseClicked(MouseEvent e) {
                if (state == GameState.SET_TABLE && tableCorners.size() < 4) {
                    tableCorners.add(new Point(e.getX(), e.getY()));
                    System.out.printf("Corner %d: x=%d y=%d%n",
                        tableCorners.size(), e.getX(), e.getY());
                    if (tableCorners.size() == 4) {
                        buildHomography();
                        tableSet = true;
                        countdownStart = System.currentTimeMillis();
                        state = GameState.COUNTDOWN;
                        System.out.println("Homography built. Countdown starting!");
                    }
                }
            }
        });

        state = GameState.REGISTER_BALL;
        System.out.println("Hold the ball in front of the camera to register...");

        // ════════════════════════════════════════════════════════════════
        // ── MAIN LOOP ────────────────────────────────────────────────────

        while (true) {

            // ── Key actions ───────────────────────────────────────────────
            if (anyKeyPressed && state == GameState.CONTROLS_SCREEN) {
                anyKeyPressed = false;
                state = GameState.REGISTER_BALL;
                System.out.println("Hold the ball in front of the camera to register...");
            }

            if (cameraSettingsRequested) {
                cameraSettingsRequested = false;
                GameState prevState = state;
                state = GameState.PAUSED;
                showCameraSettingsDialog(webcam);
                state = prevState;
            }

            if (restartRequested) {
                restartRequested = false;
                if (state == GameState.PLAYING || state == GameState.GAME_OVER
                        || state == GameState.PAUSED) {
                    resetSet();
                }
            }

            if (pauseRequested) {
                pauseRequested = false;
                if (state == GameState.PLAYING) {
                    stateBeforePause = state;
                    state = GameState.PAUSED;
                    System.out.println("Paused");
                } else if (state == GameState.PAUSED) {
                    state = GameState.PLAYING;
                    System.out.println("Resumed");
                }
            }

            if (reregisterRequested) {
                reregisterRequested = false;
                if (state == GameState.PLAYING || state == GameState.PAUSED) {
                    ballSamples.clear();
                    registeredMinRadius = 0; registeredMaxRadius = 0;
                    smoothedCenter = null;
                    state = GameState.REGISTER_BALL;
                    System.out.println("Re-registering ball...");
                }
            }

            if (undoRequested) { undoRequested = false; undoLastPoint(); }

            if (p1ScoreAdjust != 0) {
                int adj = p1ScoreAdjust; p1ScoreAdjust = 0;
                scoreP1 = Math.max(0, scoreP1 + adj);
                pointMessage = (adj > 0 ? "+1 " : "-1 ") + player1Name;
                pointMessageTime = System.currentTimeMillis();
                System.out.println("Manual P1 -> " + scoreP1);
            }
            if (p2ScoreAdjust != 0) {
                int adj = p2ScoreAdjust; p2ScoreAdjust = 0;
                scoreP2 = Math.max(0, scoreP2 + adj);
                pointMessage = (adj > 0 ? "+1 " : "-1 ") + player2Name;
                pointMessageTime = System.currentTimeMillis();
                System.out.println("Manual P2 -> " + scoreP2);
            }

            // ── Grab frame ────────────────────────────────────────────────
            BufferedImage buffered = webcam.getImage();
            if (buffered == null) { Thread.sleep(10); continue; }
            Mat frame = bufferedImageToMat(buffered);

            // Apply camera adjustments (brightness, contrast, saturation, warmth)
            applyImageAdjustments(frame);

            // ── Detect ball ───────────────────────────────────────────────
            BallResult detection = (state == GameState.PAUSED) ? null : detectBall(frame);

            if (detection != null) {
                smoothedCenter = (smoothedCenter == null) ? detection.center
                    : new Point(
                        ALPHA * detection.center.x + (1 - ALPHA) * smoothedCenter.x,
                        ALPHA * detection.center.y + (1 - ALPHA) * smoothedCenter.y);
                smoothedRadius = (smoothedRadius == 0) ? detection.radius
                    : ALPHA * detection.radius + (1 - ALPHA) * smoothedRadius;
            }

            // ════════════════════════════════════════════════════════════
            // ── State rendering ──────────────────────────────────────────

            if (state == GameState.CONTROLS_SCREEN) {
                // ── CONTROLS SCREEN ───────────────────────────────────────
                // Dark background
                Imgproc.rectangle(frame, new Point(0, 0),
                    new Point(frame.cols(), frame.rows()), new Scalar(15, 15, 15), -1);

                // Title
                drawText(frame, "TABLE TENNIS TRACKER",
                    frame.cols() / 2 - 200, 40, 0.9, new Scalar(0, 215, 255));
                drawText(frame, "Controls & How to Play",
                    frame.cols() / 2 - 145, 68, 0.65, new Scalar(180, 180, 180));

                // Divider line
                Imgproc.line(frame,
                    new Point(20, 78), new Point(frame.cols() - 20, 78),
                    new Scalar(60, 60, 60), 1);

                // Setup steps (left column)
                int lx = 25, rx = frame.cols() / 2 + 15, y = 105, gap = 26;
                drawText(frame, "SETUP STEPS",
                    lx, y - 18, 0.55, new Scalar(0, 255, 255));
                drawText(frame, "1. Hold ball to camera (30 frames)",
                    lx, y,       0.5, new Scalar(220, 220, 220));
                drawText(frame, "2. Click TL -> TR -> BR -> BL corners",
                    lx, y + gap, 0.5, new Scalar(220, 220, 220));
                drawText(frame, "3. Wait for 3-2-1 countdown",
                    lx, y + gap * 2, 0.5, new Scalar(220, 220, 220));
                drawText(frame, "4. Play! Score updates automatically",
                    lx, y + gap * 3, 0.5, new Scalar(220, 220, 220));

                // Keyboard controls (right column)
                drawText(frame, "KEYBOARD CONTROLS",
                    rx, y - 18, 0.55, new Scalar(0, 255, 255));

                String[][] keys = {
                    {"P",     "Pause / Resume game"},
                    {"R",     "Restart current set"},
                    {"Z",     "Undo last point"},
                    {"B",     "Re-register ball (if lost)"},
                    {"C",     "Camera settings (brightness etc.)"},
                    {"[  ]",  "P1 score  -1 / +1"},
                    {"-  =",  "P2 score  -1 / +1"},
                };
                for (int i = 0; i < keys.length; i++) {
                    // Key badge
                    int bx = rx, by = y + gap * i;
                    Imgproc.rectangle(frame,
                        new Point(bx, by - 14),
                        new Point(bx + 38, by + 4),
                        new Scalar(50, 50, 50), -1);
                    Imgproc.rectangle(frame,
                        new Point(bx, by - 14),
                        new Point(bx + 38, by + 4),
                        new Scalar(100, 100, 100), 1);
                    drawText(frame, keys[i][0], bx + 4, by, 0.45, new Scalar(0, 215, 255));
                    // Description
                    drawText(frame, keys[i][1], bx + 46, by, 0.48, new Scalar(210, 210, 210));
                }

                // Scoring rules section
                int ry = y + gap * 7;
                Imgproc.line(frame,
                    new Point(20, ry - 12), new Point(frame.cols() - 20, ry - 12),
                    new Scalar(60, 60, 60), 1);
                drawText(frame, "SCORING RULES",
                    lx, ry + 4, 0.55, new Scalar(0, 255, 255));
                drawText(frame, "Ball bounces twice on same side  ->  opponent scores",
                    lx, ry + gap, 0.47, new Scalar(200, 200, 200));
                drawText(frame, "Ball leaves table boundary       ->  opponent scores",
                    lx, ry + gap * 2, 0.47, new Scalar(200, 200, 200));
                drawText(frame, "Ball missing 25+ frames          ->  opponent scores",
                    lx, ry + gap * 3, 0.47, new Scalar(200, 200, 200));
                drawText(frame, "Serve switches every 2 points   |   Win by 2 at deuce",
                    lx, ry + gap * 4, 0.47, new Scalar(200, 200, 200));

                // Prompt to continue
                long pulse = (System.currentTimeMillis() / 600) % 2 == 0
                    ? 255 : 160;
                drawText(frame, "Press any key to begin",
                    frame.cols() / 2 - 120, frame.rows() - 18,
                    0.65, new Scalar(0, pulse, pulse));

            } else if (state == GameState.REGISTER_BALL) {
                // ── REGISTER BALL ─────────────────────────────────────────
                drawText(frame, "Hold ball close to camera to register",
                    10, 30, 0.65, new Scalar(0, 255, 255));
                drawText(frame, "Samples: " + ballSamples.size() + "/" + SAMPLE_COUNT,
                    10, 60, 0.65, new Scalar(255, 255, 0));
                drawText(frame, "Press B anytime during game to re-register",
                    10, 90, 0.45, new Scalar(180, 180, 180));

                if (detection != null && detection.radius > 5) {
                    ballSamples.add(detection.radius);
                    Imgproc.circle(frame, detection.center, (int) detection.radius,
                        new Scalar(0, 255, 0), 2);
                    drawText(frame, "r=" + String.format("%.1f", detection.radius),
                        10, 115, 0.6, new Scalar(0, 255, 0));
                }

                if (ballSamples.size() >= SAMPLE_COUNT) {
                    double sum = 0;
                    for (double s : ballSamples) sum += s;
                    double avgR = sum / ballSamples.size();
                    registeredMinRadius = avgR * 0.5;
                    registeredMaxRadius = avgR * 1.5;
                    System.out.printf("Ball registered! avg=%.1f range=[%.1f-%.1f]%n",
                        avgR, registeredMinRadius, registeredMaxRadius);
                    if (tableSet) {
                        countdownStart = System.currentTimeMillis();
                        state = GameState.COUNTDOWN;
                    } else {
                        state = GameState.SET_TABLE;
                        System.out.println("Click 4 corners: TL -> TR -> BR -> BL");
                    }
                }

            } else if (state == GameState.SET_TABLE) {
                // ── SET TABLE ─────────────────────────────────────────────
                drawText(frame, "Click 4 corners: TL -> TR -> BR -> BL",
                    10, 30, 0.6, new Scalar(255, 255, 0));
                drawText(frame, "Corners: " + tableCorners.size() + "/4",
                    10, 60, 0.6, new Scalar(0, 255, 255));
                String[] labels = {"TL", "TR", "BR", "BL"};
                for (int i = 0; i < tableCorners.size(); i++) {
                    Imgproc.circle(frame, tableCorners.get(i), 6,
                        new Scalar(0, 255, 255), -1);
                    drawText(frame, labels[i],
                        (int) tableCorners.get(i).x + 8,
                        (int) tableCorners.get(i).y,
                        0.5, new Scalar(0, 255, 255));
                }

            } else if (state == GameState.COUNTDOWN) {
                // ── COUNTDOWN ─────────────────────────────────────────────
                drawTableOverlay(frame);
                long elapsed   = (System.currentTimeMillis() - countdownStart) / 1000;
                int  remaining = COUNTDOWN_SECS - (int) elapsed;

                if (remaining > 0) {
                    drawText(frame, String.valueOf(remaining),
                        frame.cols() / 2 - 25, frame.rows() / 2 + 40,
                        3.0, new Scalar(0, 255, 255));
                    drawText(frame, "Get Ready!",
                        frame.cols() / 2 - 90, frame.rows() / 2 - 20,
                        1.0, new Scalar(255, 255, 255));
                } else {
                    drawText(frame, "GO!",
                        frame.cols() / 2 - 45, frame.rows() / 2 + 40,
                        2.5, new Scalar(0, 255, 0));
                    if (elapsed >= COUNTDOWN_SECS + 1) {
                        state = GameState.PLAYING;
                        System.out.println("Game started!");
                    }
                }

            } else if (state == GameState.PAUSED) {
                // ── PAUSED ────────────────────────────────────────────────
                drawTableOverlay(frame);
                drawScoreboard(frame);
                Imgproc.rectangle(frame,
                    new Point(frame.cols() / 2 - 130, frame.rows() / 2 - 35),
                    new Point(frame.cols() / 2 + 130, frame.rows() / 2 + 65),
                    new Scalar(10, 10, 10), -1);
                drawText(frame, "PAUSED",
                    frame.cols() / 2 - 80, frame.rows() / 2 + 15,
                    1.5, new Scalar(0, 255, 255));
                drawText(frame, "Press P to resume",
                    frame.cols() / 2 - 105, frame.rows() / 2 + 50,
                    0.65, new Scalar(200, 200, 200));

            } else if (state == GameState.PLAYING || state == GameState.GAME_OVER) {
                // ── PLAYING / GAME OVER ───────────────────────────────────
                drawTableOverlay(frame);

                if (state == GameState.PLAYING) {
                    long now = System.currentTimeMillis();

                    Point flatBall = null;
                    if (smoothedCenter != null) {
                        flatBall = toFlatSpace(smoothedCenter);
                    } else if (!posHistory.isEmpty()) {
                        flatBall = predictNextPosition();
                        predictedPos = flatBall;
                    }

                    if (smoothedCenter != null && detection != null) {
                        ballMissingFrames = 0;
                        predictedPos = null;

                        Imgproc.circle(frame, smoothedCenter, (int) smoothedRadius,
                            new Scalar(0, 255, 0), 2);
                        Imgproc.circle(frame, smoothedCenter, 3,
                            new Scalar(0, 0, 255), -1);

                        if (flatBall != null) {
                            // Speed calculation
                            if (!posHistory.isEmpty()) {
                                Point lastPos = posHistory.getLast();
                                double dpx = Math.hypot(
                                    flatBall.x - lastPos.x, flatBall.y - lastPos.y);
                                ballSpeedKmh = dpx * (TABLE_LENGTH_M / FLAT_W) * FPS * 3.6;
                            }

                            posHistory.addLast(flatBall);
                            if (posHistory.size() > TRAJ_HISTORY) posHistory.removeFirst();
                            drawTrajectory(frame);

                            boolean onTable =
                                flatBall.x >= 0 && flatBall.x <= FLAT_W &&
                                flatBall.y >= 0 && flatBall.y <= FLAT_H;

                            if (onTable) {
                                boolean onP1Side   = flatBall.x < FLAT_W / 2.0;
                                String  currentSide = onP1Side ? "P1" : "P2";

                                // Net crossing
                                if (ballSeenOnce && onP1Side != wasOnP1Side) {
                                    stats.netCrossing();
                                    // Let detection: crossed and came back very quickly
                                    if (justCrossed && netCrossFrames <= 5) {
                                        pointMessage     = "LET - Replay!";
                                        pointMessageTime = now;
                                        justCrossed = false;
                                        resetBallTracking();
                                    } else {
                                        justCrossed    = true;
                                        netCrossFrames = 0;
                                    }
                                }
                                if (justCrossed) netCrossFrames++;
                                if (netCrossFrames > 10) justCrossed = false;

                                wasOnP1Side  = onP1Side;
                                ballSeenOnce = true;
                                lastSide     = currentSide;

                                // Bounce detection
                                if (prevFlatY >= 0) {
                                    double dy = flatBall.y - prevFlatY;
                                    if (prevDY > 1.5 && dy < -1.5) {
                                        if (bounceSide.equals(currentSide)) {
                                            bounceCount++;
                                        } else {
                                            bounceSide  = currentSide;
                                            bounceCount = 1;
                                        }
                                        System.out.printf("Bounce #%d on %s%n",
                                            bounceCount, currentSide);
                                        if (bounceCount >= BOUNCE_THRESHOLD
                                                && now - lastPointTime > 1500) {
                                            awardPoint(currentSide.equals("P1") ? "P2" : "P1");
                                            lastPointTime = pointMessageTime = now;
                                            resetBallTracking();
                                        }
                                    }
                                    prevDY = dy;
                                }
                                prevFlatY = flatBall.y;

                            } else {
                                // Ball left table
                                if (!lastSide.isEmpty() && now - lastPointTime > 1500) {
                                    awardPoint(lastSide.equals("P1") ? "P2" : "P1");
                                    lastPointTime = pointMessageTime = now;
                                    resetBallTracking();
                                }
                            }
                        }

                    } else {
                        // Ball not visible — show ghost
                        ballMissingFrames++;
                        if (predictedPos != null) {
                            Point rawPred = toRawSpace(predictedPos);
                            if (rawPred != null) {
                                Imgproc.circle(frame, rawPred, (int) smoothedRadius,
                                    new Scalar(0, 200, 200), 1);
                                drawText(frame, "?",
                                    (int) rawPred.x - 5, (int) rawPred.y + 5,
                                    0.5, new Scalar(0, 200, 200));
                            }
                        }
                        if (ballMissingFrames >= MISSING_THRESHOLD
                                && !lastSide.isEmpty() && now - lastPointTime > 1500) {
                            awardPoint(lastSide.equals("P1") ? "P2" : "P1");
                            lastPointTime = pointMessageTime = now;
                            resetBallTracking();
                        }
                    }

                    checkWinCondition(now);
                }

                drawScoreboard(frame);

                // Ball speed display
                if (ballSpeedKmh > 0.5 && state == GameState.PLAYING) {
                    drawText(frame,
                        String.format("%.1f km/h", ballSpeedKmh),
                        frame.cols() - 115, frame.rows() - 30,
                        0.6, new Scalar(180, 255, 180));
                }

                // Point flash message
                long now2 = System.currentTimeMillis();
                if (!pointMessage.isEmpty() && now2 - pointMessageTime < 2000) {
                    // Shadow
                    drawText(frame, pointMessage,
                        frame.cols() / 2 - 162, frame.rows() / 2 + 2,
                        1.0, new Scalar(0, 0, 0));
                    drawText(frame, pointMessage,
                        frame.cols() / 2 - 160, frame.rows() / 2,
                        1.0, new Scalar(0, 255, 255));
                }

                // Controls hint bar
                drawText(frame,
                    "P=Pause  B=Re-reg  C=Camera  Z=Undo  R=Restart  [/]=P1  -/==P2",
                    5, frame.rows() - 8, 0.37, new Scalar(130, 130, 130));

                // ── Game Over overlay ─────────────────────────────────────
                if (state == GameState.GAME_OVER) {
                    String setWinner   = scoreP1 > scoreP2 ? player1Name : player2Name;
                    String matchWinner = setsP1 >= setsToWin ? player1Name
                                       : setsP2 >= setsToWin ? player2Name : "";

                    Imgproc.rectangle(frame,
                        new Point(0, frame.rows() / 2 - 10),
                        new Point(frame.cols(), frame.rows() - 22),
                        new Scalar(10, 10, 10), -1);

                    if (!matchWinner.isEmpty()) {
                        drawText(frame, matchWinner + " WINS THE MATCH!",
                            frame.cols() / 2 - 220, frame.rows() / 2 + 28,
                            1.05, new Scalar(0, 215, 255));
                    } else {
                        drawText(frame, setWinner + " wins Set " + (setsP1 + setsP2),
                            frame.cols() / 2 - 195, frame.rows() / 2 + 28,
                            1.0, new Scalar(0, 215, 255));
                        drawText(frame,
                            "Sets: " + player1Name + " " + setsP1
                                + "  -  " + setsP2 + " " + player2Name,
                            frame.cols() / 2 - 165, frame.rows() / 2 + 52,
                            0.6, new Scalar(200, 200, 200));
                        drawText(frame, "Press R for next set",
                            frame.cols() / 2 - 110, frame.rows() / 2 + 74,
                            0.6, new Scalar(200, 200, 200));
                    }

                    int sy = frame.rows() / 2 + 90;
                    drawText(frame, "Total points: " + stats.totalPointsPlayed,
                        30, sy,      0.5, new Scalar(220, 220, 220));
                    drawText(frame, "Longest rally: " + stats.longestRally + " crossings",
                        30, sy + 20, 0.5, new Scalar(220, 220, 220));
                    drawText(frame, player1Name + " best streak: " + stats.p1MaxConsecutive,
                        30, sy + 40, 0.5, new Scalar(100, 200, 255));
                    drawText(frame, player2Name + " best streak: " + stats.p2MaxConsecutive,
                        30, sy + 60, 0.5, new Scalar(100, 255, 100));
                    drawText(frame,
                        String.format("Avg rally: %.1f crossings", stats.avgRallyLength()),
                        30, sy + 80, 0.5, new Scalar(200, 200, 100));
                }
            }

            // ── Render frame ──────────────────────────────────────────────
            final ImageIcon icon = new ImageIcon(matToBufferedImage(frame));
            SwingUtilities.invokeLater(() -> {
                imageLabel.setIcon(icon);
                window.repaint();
            });

            Thread.sleep(30);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Camera Settings Dialog ───────────────────────────────────────────

    static void showCameraSettingsDialog(Webcam webcam) throws InterruptedException {
        // Sliders
        JSlider brightSlider = makeSlider(-100, 100,  0);
        JSlider contrastSlider = makeSlider(50,  200, 100);  // stored as x/100
        JSlider saturSlider  = makeSlider(0,    200, 100);   // stored as x/100
        JSlider warmthSlider = makeSlider(-50,   50,  0);

        // Live preview label
        JLabel preview = new JLabel();
        preview.setPreferredSize(new java.awt.Dimension(480, 270));

        // Layout
        JPanel controls = new JPanel();
        controls.setLayout(new BoxLayout(controls, BoxLayout.Y_AXIS));
        controls.add(labeledRow("Brightness",   brightSlider));
        controls.add(labeledRow("Contrast",     contrastSlider));
        controls.add(labeledRow("Saturation",   saturSlider));
        controls.add(labeledRow("Warmth (White Balance)", warmthSlider));
        controls.add(Box.createVerticalStrut(6));

        JButton resetBtn = new JButton("Reset to Default");
        resetBtn.setAlignmentX(java.awt.Component.CENTER_ALIGNMENT);
        resetBtn.addActionListener(e -> {
            brightSlider.setValue(0);
            contrastSlider.setValue(100);
            saturSlider.setValue(100);
            warmthSlider.setValue(0);
        });
        controls.add(resetBtn);

        JPanel dialogPanel = new JPanel(new java.awt.BorderLayout(10, 10));
        dialogPanel.add(preview,  java.awt.BorderLayout.WEST);
        dialogPanel.add(controls, java.awt.BorderLayout.CENTER);

        // Update preview when sliders move
        javax.swing.event.ChangeListener cl = e -> {
            camBrightness = brightSlider.getValue();
            camContrast   = contrastSlider.getValue() / 100.0;
            camSaturation = saturSlider.getValue()  / 100.0;
            camWarmth     = warmthSlider.getValue();

            BufferedImage raw = webcam.getImage();
            if (raw == null) return;
            Mat frame = bufferedImageToMat(raw);
            applyImageAdjustments(frame);
            BufferedImage adj = matToBufferedImage(frame);
            // Scale down for preview
            java.awt.Image scaled = adj.getScaledInstance(480, 270, java.awt.Image.SCALE_FAST);
            preview.setIcon(new ImageIcon(scaled));
        };

        brightSlider.addChangeListener(cl);
        contrastSlider.addChangeListener(cl);
        saturSlider.addChangeListener(cl);
        warmthSlider.addChangeListener(cl);

        // Grab one initial frame for preview
        Thread.sleep(200);
        BufferedImage raw = webcam.getImage();
        if (raw != null) {
            java.awt.Image scaled = raw.getScaledInstance(480, 270, java.awt.Image.SCALE_FAST);
            preview.setIcon(new ImageIcon(scaled));
        }

        JOptionPane.showConfirmDialog(null, dialogPanel,
            "Camera Settings  (adjust until the table looks clear)",
            JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

        // Commit final values
        camBrightness = brightSlider.getValue();
        camContrast   = contrastSlider.getValue() / 100.0;
        camSaturation = saturSlider.getValue()  / 100.0;
        camWarmth     = warmthSlider.getValue();
        System.out.printf("Camera settings: brightness=%.0f contrast=%.2f " +
            "saturation=%.2f warmth=%.0f%n",
            camBrightness, camContrast, camSaturation, camWarmth);
    }

    /** Build a horizontal slider with tick marks. */
    static JSlider makeSlider(int min, int max, int val) {
        JSlider s = new JSlider(min, max, val);
        s.setPaintTicks(true);
        s.setPaintLabels(false);
        s.setMajorTickSpacing((max - min) / 4);
        return s;
    }

    /** Wrap a slider with a label showing its name and current value. */
    static JPanel labeledRow(String name, JSlider slider) {
        JLabel lbl = new JLabel(String.format("%-28s %+d", name, slider.getValue()));
        lbl.setPreferredSize(new java.awt.Dimension(260, 18));
        slider.addChangeListener(e ->
            lbl.setText(String.format("%-28s %+d", name, slider.getValue())));
        JPanel row = new JPanel(new java.awt.BorderLayout(6, 0));
        row.add(lbl,    java.awt.BorderLayout.WEST);
        row.add(slider, java.awt.BorderLayout.CENTER);
        row.setMaximumSize(new java.awt.Dimension(9999, 40));
        return row;
    }

    /**
     * Apply brightness, contrast, saturation, and warmth to a BGR Mat in place.
     *
     *  Brightness : adds a constant to all channels
     *  Contrast   : multiplies all channels by a factor (alpha in convertTo)
     *  Saturation : adjusts S channel in HSV
     *  Warmth     : boosts/reduces R and B channels (warm = more R, cool = more B)
     */
    static void applyImageAdjustments(Mat frame) {
        // Brightness + Contrast  (dst = src * contrast + brightness)
        frame.convertTo(frame, -1, camContrast, camBrightness);

        // Saturation (work in HSV)
        if (Math.abs(camSaturation - 1.0) > 0.01) {
            Mat hsv = new Mat();
            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
            List<Mat> ch = new ArrayList<>();
            Core.split(hsv, ch);
            ch.get(1).convertTo(ch.get(1), -1, camSaturation, 0);
            Core.merge(ch, hsv);
            Imgproc.cvtColor(hsv, frame, Imgproc.COLOR_HSV2BGR);
        }

        // Warmth: shift red and blue channels
        if (Math.abs(camWarmth) > 0.5) {
            List<Mat> ch = new ArrayList<>();
            Core.split(frame, ch);                     // ch: B, G, R
            double warm = camWarmth;
            ch.get(2).convertTo(ch.get(2), -1, 1.0,  warm);   // R up = warm
            ch.get(0).convertTo(ch.get(0), -1, 1.0, -warm);   // B down = warm
            Core.merge(ch, frame);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Scoreboard ───────────────────────────────────────────────────────

    static void drawScoreboard(Mat frame) {
        Imgproc.rectangle(frame, new Point(0, 0),
            new Point(frame.cols(), 62), new Scalar(20, 20, 20), -1);

        // Serve indicator: yellow = serving
        Scalar p1col = servingPlayer.equals("P1") ? new Scalar(0, 255, 255) : new Scalar(100, 200, 255);
        Scalar p2col = servingPlayer.equals("P2") ? new Scalar(0, 255, 255) : new Scalar(100, 255, 100);

        String p1text = (servingPlayer.equals("P1") ? ">> " : "   ") + player1Name + ": " + scoreP1;
        String p2text = player2Name + ": " + scoreP2 + (servingPlayer.equals("P2") ? " <<" : "");

        drawText(frame, p1text, 10, 40, 0.9, p1col);
        drawText(frame, "VS", frame.cols() / 2 - 18, 38, 0.8, new Scalar(255, 255, 255));

        int[] base = new int[1];
        Size  sz   = Imgproc.getTextSize(p2text, Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, 2, base);
        drawText(frame, p2text, (int)(frame.cols() - sz.width - 10), 40, 0.9, p2col);

        // Sets score
        String setsStr = "Sets " + setsP1 + " - " + setsP2;
        drawText(frame, setsStr, frame.cols() / 2 - 35, 58, 0.42, new Scalar(180, 180, 180));

        // Rally + serve info
        String serveLabel = (servingPlayer.equals("P1") ? player1Name : player2Name) + " serving";
        drawText(frame, serveLabel, 10, 58, 0.42, new Scalar(0, 200, 200));
        drawText(frame, "Rally: " + stats.currentRally,
            frame.cols() - 100, 58, 0.42, new Scalar(180, 180, 180));
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Table overlay ────────────────────────────────────────────────────

    static void drawTableOverlay(Mat frame) {
        if (tableCorners.size() < 4) return;
        Point tl = tableCorners.get(0), tr = tableCorners.get(1);
        Point br = tableCorners.get(2), bl = tableCorners.get(3);
        Imgproc.line(frame, tl, tr, new Scalar(255, 80, 0), 2);
        Imgproc.line(frame, tr, br, new Scalar(255, 80, 0), 2);
        Imgproc.line(frame, br, bl, new Scalar(255, 80, 0), 2);
        Imgproc.line(frame, bl, tl, new Scalar(255, 80, 0), 2);
        Point netTop = midpoint(tl, tr), netBot = midpoint(bl, br);
        Imgproc.line(frame, netTop, netBot, new Scalar(0, 0, 255), 2);
        drawText(frame, player1Name,
            (int) tl.x + 10, (int) tl.y + 75, 0.55, new Scalar(100, 200, 255));
        drawText(frame, player2Name,
            (int) netTop.x + 10, (int) netTop.y + 75, 0.55, new Scalar(100, 255, 100));
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Win condition ────────────────────────────────────────────────────

    static void checkWinCondition(long now) {
        boolean atDeuce = scoreP1 >= 10 && scoreP2 >= 10;
        boolean p1Won   = atDeuce ? (scoreP1 - scoreP2 >= 2) : (scoreP1 >= 11);
        boolean p2Won   = atDeuce ? (scoreP2 - scoreP1 >= 2) : (scoreP2 >= 11);

        if (p1Won) { setsP1++; state = GameState.GAME_OVER;
            System.out.printf("Set over! Sets: %s=%d %s=%d%n", player1Name, setsP1, player2Name, setsP2); }
        else if (p2Won) { setsP2++; state = GameState.GAME_OVER;
            System.out.printf("Set over! Sets: %s=%d %s=%d%n", player1Name, setsP1, player2Name, setsP2); }

        if (atDeuce && !p1Won && !p2Won) {
            String label = (scoreP1 == scoreP2) ? "DEUCE"
                : (scoreP1 > scoreP2) ? "ADV " + player1Name : "ADV " + player2Name;
            pointMessage = label; pointMessageTime = now;
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Homography ───────────────────────────────────────────────────────

    static void buildHomography() {
        MatOfPoint2f src = new MatOfPoint2f(
            tableCorners.get(0), tableCorners.get(1),
            tableCorners.get(2), tableCorners.get(3));
        MatOfPoint2f dst = new MatOfPoint2f(
            new Point(0, 0), new Point(FLAT_W, 0),
            new Point(FLAT_W, FLAT_H), new Point(0, FLAT_H));
        homographyMatrix  = Imgproc.getPerspectiveTransform(src, dst);
        homographyInverse = Imgproc.getPerspectiveTransform(dst, src);
    }

    static Point toFlatSpace(Point p) {
        if (homographyMatrix == null) return p;
        MatOfPoint2f src = new MatOfPoint2f(p), dst = new MatOfPoint2f();
        Core.perspectiveTransform(src, dst, homographyMatrix);
        return dst.toArray()[0];
    }

    static Point toRawSpace(Point p) {
        if (homographyInverse == null) return p;
        MatOfPoint2f src = new MatOfPoint2f(p), dst = new MatOfPoint2f();
        Core.perspectiveTransform(src, dst, homographyInverse);
        return dst.toArray()[0];
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Trajectory ───────────────────────────────────────────────────────

    static Point predictNextPosition() {
        if (posHistory.size() < 2) return null;
        Point[] a = posHistory.toArray(new Point[0]);
        Point last = a[a.length - 1], prev = a[a.length - 2];
        return new Point(last.x + (last.x - prev.x), last.y + (last.y - prev.y));
    }

    static void drawTrajectory(Mat frame) {
        if (posHistory.size() < 2) return;
        Point[] a = posHistory.toArray(new Point[0]);
        Point last = a[a.length - 1], prev = a[a.length - 2];
        double dx = last.x - prev.x, dy = last.y - prev.y;
        for (int i = 1; i <= 4; i++) {
            Point rp = toRawSpace(new Point(last.x + dx * i, last.y + dy * i));
            if (rp != null) Imgproc.circle(frame, rp, 3,
                new Scalar(0, Math.max(0, 200 - i * 40), Math.max(0, 200 - i * 40)), -1);
        }
        for (int i = 1; i < a.length; i++) {
            Point ra = toRawSpace(a[i - 1]), rb = toRawSpace(a[i]);
            if (ra != null && rb != null)
                Imgproc.line(frame, ra, rb, new Scalar(0, 180, 0), 1);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Scoring helpers ──────────────────────────────────────────────────

    static void awardPoint(String scorer) {
        if (scorer.equals("P1")) { scoreP1++; pointMessage = "Point -> " + player1Name + "!"; }
        else                     { scoreP2++; pointMessage = "Point -> " + player2Name + "!"; }

        PointRecord rec = new PointRecord(scorer, scoreP1, scoreP2, stats.currentRally);
        undoStack.addLast(rec);
        if (undoStack.size() > 10) undoStack.removeFirst();
        matchLog.add(rec);
        writeLogLine(rec);
        stats.recordPoint(scorer);

        totalPointsDone++;
        if (totalPointsDone % 2 == 0) {
            servingPlayer = servingPlayer.equals("P1") ? "P2" : "P1";
            System.out.println("Serve -> "
                + (servingPlayer.equals("P1") ? player1Name : player2Name));
        }
        System.out.printf("Point! %s=%d  %s=%d  Serving=%s%n",
            player1Name, scoreP1, player2Name, scoreP2, servingPlayer);
    }

    static void undoLastPoint() {
        if (undoStack.isEmpty()) { System.out.println("Nothing to undo."); return; }
        PointRecord last = undoStack.removeLast();
        if (last.scorer.equals("P1")) scoreP1 = Math.max(0, scoreP1 - 1);
        else                          scoreP2 = Math.max(0, scoreP2 - 1);
        totalPointsDone = Math.max(0, totalPointsDone - 1);
        if ((totalPointsDone + 1) % 2 == 0)
            servingPlayer = servingPlayer.equals("P1") ? "P2" : "P1";
        stats.undoPoint(last.scorer);
        pointMessage     = "Undone!";
        pointMessageTime = System.currentTimeMillis();
        if (state == GameState.GAME_OVER) state = GameState.PLAYING;
        System.out.printf("Undo! %s=%d  %s=%d%n", player1Name, scoreP1, player2Name, scoreP2);
    }

    static void resetSet() {
        scoreP1 = 0; scoreP2 = 0; pointMessage = "";
        totalPointsDone = 0; servingPlayer = "P1";
        stats.totalPointsPlayed = 0; stats.longestRally = 0;
        stats.currentRally = 0; stats.p1MaxConsecutive = 0;
        stats.p2MaxConsecutive = 0; stats.p1Consecutive = 0;
        stats.p2Consecutive = 0; stats.totalRallyFrames = 0; stats.rallyCount = 0;
        undoStack.clear();
        resetBallTracking();
        countdownStart = System.currentTimeMillis();
        state = GameState.COUNTDOWN;
        System.out.println("New set!");
    }

    static void resetBallTracking() {
        lastSide = ""; smoothedCenter = null; ballMissingFrames = 0;
        posHistory.clear(); predictedPos = null;
        prevFlatY = -1; prevDY = 0; bounceSide = ""; bounceCount = 0;
        ballSeenOnce = false; justCrossed = false; netCrossFrames = 0;
        ballSpeedKmh = 0;
    }

    static void writeLogLine(PointRecord rec) {
        if (logWriter == null) return;
        logWriter.printf("%s,%s,%d,%d,%d%n",
            rec.timestamp, rec.scorer, rec.p1score, rec.p2score, rec.rallyLength);
        logWriter.flush();
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Ball detection ───────────────────────────────────────────────────

    static BallResult detectBall(Mat frame) {
        Mat hsv = new Mat(), mask = new Mat();
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsv, ballLower, ballUpper, mask);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.erode(mask, mask, kernel);
        Imgproc.dilate(mask, mask, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(),
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double bestScore = 0; Point bestCenter = null; double bestRadius = 0;
        float[] r = new float[1]; Point c = new Point();

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < 50 || area > 8000) continue;
            double perim = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            if (perim == 0) continue;
            double circ = (4 * Math.PI * area) / (perim * perim);
            if (circ < 0.55 || circ <= bestScore) continue;
            Imgproc.minEnclosingCircle(new MatOfPoint2f(contour.toArray()), c, r);
            if (registeredMinRadius > 0
                    && (r[0] < registeredMinRadius || r[0] > registeredMaxRadius)) continue;
            bestScore = circ; bestCenter = new Point(c.x, c.y); bestRadius = r[0];
        }
        return bestCenter != null ? new BallResult(bestCenter, bestRadius) : null;
    }

    // ════════════════════════════════════════════════════════════════════
    // ── Utility ──────────────────────────────────────────────────────────

    static void drawText(Mat frame, String text, int x, int y, double scale, Scalar color) {
        if (frame == null || text == null) return;
        Imgproc.putText(frame, text, new Point(x, y),
            Imgproc.FONT_HERSHEY_SIMPLEX, scale, color, 2);
    }

    static Point midpoint(Point a, Point b) {
        return new Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
    }

    static Mat bufferedImageToMat(BufferedImage img) {
        // Draw into a known TYPE_INT_RGB so we control the byte order
        BufferedImage rgb = new BufferedImage(
            img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = rgb.createGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();

        int w = rgb.getWidth(), h = rgb.getHeight();
        byte[] data = new byte[w * h * 3];
        int idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int pixel = rgb.getRGB(x, y);
                data[idx++] = (byte)((pixel      ) & 0xFF); // B  (OpenCV wants BGR)
                data[idx++] = (byte)((pixel >>  8) & 0xFF); // G
                data[idx++] = (byte)((pixel >> 16) & 0xFF); // R
            }
        }
        Mat mat = new Mat(h, w, CvType.CV_8UC3);
        mat.put(0, 0, data);
        return mat;
    }

    static BufferedImage matToBufferedImage(Mat mat) {
        int w = mat.cols(), h = mat.rows();
        byte[] data = new byte[w * h * 3];
        mat.get(0, 0, data);
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int b = data[idx++] & 0xFF;  // OpenCV BGR order
                int g = data[idx++] & 0xFF;
                int r = data[idx++] & 0xFF;
                img.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return img;
    }
}
