`timescale 1ns / 1ps
module RTF_top #(
    // basic setting
    parameter MIC_NUM              = 8,
    parameter SOR_NUM              = 2,
    parameter FREQ_NUM             = 257,

    // Data width
    parameter DATA_WIDTH           = 16,
    parameter ACC_WIDTH            = 33,
    parameter INV_G_WIDTH          = 105,
    parameter INPUT_FRAC_BITS      = 14,

    // bram read data latency
    parameter LATENCY              = 2,

    // bram address width and start point
    parameter BRAM_RD_ADDR_WIDTH   = 13,
    parameter BRAM_WR_ADDR_WIDTH   = 13,
    parameter BRAM_RD_ADDR_BASE    = 0,
    parameter BRAM_WR_ADDR_BASE    = 0,
    parameter BRAM_WR_WE_WIDTH     = 8,

    // bram address increase number
    parameter BRAM_RD_INCREASE     = 1,  
    parameter BRAM_WR_INCREASE     = 1, 

    // divider data width
    parameter DIVOUT_TDATA_WIDTH   = 72,
    parameter DIVOUT_F_WIDTH       = 24, // Vivado divider: MSB of frac is sign, so 24 (1 sign + 23 frac)
    parameter DIVOUT_GUARD_BITS    = 0,
    parameter DIVISOR_TDATA_WIDTH  = 64,
    parameter DIVIDEND_TDATA_WIDTH = 48,

    // 1 = dynamic det_shift (like software), 0 = fixed DET_SCALE_SHIFT
    parameter USE_DYNAMIC_DET_SHIFT = 1,

    // lambda number
    parameter LAMBDA                = 33'd2684355
)(
    input                                        clk,
    input                                        rst_n,
    input                                        start,
    output reg                                   done,
    output reg                                   all_freq_finish,

    // read bram data
    input      signed [DATA_WIDTH-1:0]           af_bram_rd_real,
    input      signed [DATA_WIDTH-1:0]           af_bram_rd_imag,
    output reg        [BRAM_RD_ADDR_WIDTH-1:0]   bram_rd_addr,

    // write bram data
    output reg signed [INV_G_WIDTH-1:0]          result_bram_wr_real,
    output reg signed [INV_G_WIDTH-1:0]          result_bram_wr_imag,
    output reg        [BRAM_WR_ADDR_WIDTH-1:0]   bram_wr_addr,
    output            [BRAM_WR_WE_WIDTH-1:0]     bram_wr_we,
    output                                       bram_wr_en,

    // from divider
    input      signed [DIVOUT_TDATA_WIDTH-1:0]   m_axis_dout_tdata,
    input                                        m_axis_dout_tvalid,

    // to divider
    output reg signed [DIVIDEND_TDATA_WIDTH-1:0] s_axis_dividend_tdata,
    output reg                                   s_axis_dividend_tvalid,
    output reg signed [DIVISOR_TDATA_WIDTH-1:0]  s_axis_divisor_tdata,
    output reg                                   s_axis_divisor_tvalid
);

    localparam S_IDLE           = 0;  // wait start
    localparam S_RD             = 1;  // read bram data
    localparam S_UPDATE_RD_ADDR = 2;  // update bram read address
    localparam S_WAIT_RD_DELAY  = 3;  // wait bram read delay
    localparam S_PLUS           = 4;  // plus lambda * I to G
    localparam S_CALDET1        = 5;  // calculate g11 * g22
    localparam S_CALDET2        = 6;  // calculate det - (g12_real_acc_sqr + g12_imag_acc_sqr)
    localparam S_INVDET         = 7;  // set dividend and divisor to divider
    localparam S_SETDIV         = 8;  // set dividend and divisor valid to divider
    localparam S_WAITDIV        = 9;  // wait divider result
    localparam S_CALINVG        = 10; // calculate inverse of G
    localparam S_CALRESULT      = 11; // calculate result elements
    localparam S_WR             = 12; // write result to bram
    localparam S_UPDATE_WR_ADDR = 13; // update bram write address
    localparam S_DONE           = 14; // done
    localparam S_RESTART        = 15; // return to S_RD

    localparam TOTAL_NUM          = MIC_NUM * SOR_NUM * FREQ_NUM;                                                      // 8 * 2 * 257 = 4112
    localparam PER_FREQ           = MIC_NUM * SOR_NUM;                                                                 // 8 * 2 = 16
    localparam DET_WIDTH          = ACC_WIDTH * 2;                                                                     // 33 * 2 = 66
    localparam DET_FRAC_BITS      = 4 * INPUT_FRAC_BITS;                                                               // 14 * 4 = 56
    localparam DIVOUT_Q_WIDTH     = DIVOUT_TDATA_WIDTH - DIVOUT_F_WIDTH;                                               // 72 - 24 = 48
    localparam DIVIDEND_SHIFT     = (DIVOUT_F_WIDTH - 1) + DET_FRAC_BITS + DIVOUT_GUARD_BITS;                          // 15 + 56 + 0 = 71
    localparam DIVIDEND_MAX_SHIFT = DIVIDEND_TDATA_WIDTH - 1;                                                          // 47
    localparam DET_SCALE_SHIFT    = (DIVIDEND_SHIFT > DIVIDEND_MAX_SHIFT) ? (DIVIDEND_SHIFT - DIVIDEND_MAX_SHIFT) : 0; // 71 - 47 = 24
    localparam TARGET_BITS        = DIVOUT_F_WIDTH + 1;                                                                // software: det_bits - this
    localparam [6:0] DET_SCALE_SHIFT_7 = DET_SCALE_SHIFT;

    // ==============================
    // bram start delay
    // ==============================
    reg [LATENCY:0] start_delay;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_delay <= 0;
        end else begin
            start_delay <= {start_delay[LATENCY-1:0], start};
        end
    end

    // ==============================
    // FSM setting
    // ==============================
    reg [3:0] state;
    reg [3:0] next_state;
    reg [2:0] sor_cnt;
    reg [3:0] rd_cnt;
    reg [3:0] wr_cnt;
    reg       flag_rd_sor1;
    reg       result_row1;
    reg [8:0] freq_sample_cnt;
    reg       wait_rd_change; 

    assign bram_wr_we = (state == S_WR) ? {BRAM_WR_WE_WIDTH{1'b1}} : {BRAM_WR_WE_WIDTH{1'b0}};
    assign bram_wr_en = (state == S_WR);
    

    reg signed [DATA_WIDTH-1:0] sor0_temp_real [0:MIC_NUM-1];
    reg signed [DATA_WIDTH-1:0] sor0_temp_imag [0:MIC_NUM-1];
    reg signed [DATA_WIDTH-1:0] sor1_temp_real [0:MIC_NUM-1];
    reg signed [DATA_WIDTH-1:0] sor1_temp_imag [0:MIC_NUM-1];

    // G = a(f)h * a(f) + lambda * I register
    // note1: g21 = g12 conjugate, so we only need store g11, g12, g22.
    // note2: g11 and g22 only have real part, so we only need store real part of g11 and g22
    reg signed [ACC_WIDTH-1:0] g11_real_acc;
    reg signed [ACC_WIDTH-1:0] g12_real_acc;
    reg signed [ACC_WIDTH-1:0] g12_imag_acc;
    reg signed [ACC_WIDTH-1:0] g22_real_acc;

    // square of g12 (avoid timing violation)
    wire signed [DET_WIDTH-1:0] g12_real_acc_sqr;
    wire signed [DET_WIDTH-1:0] g12_imag_acc_sqr;

    assign g12_real_acc_sqr = g12_real_acc * g12_real_acc;
    assign g12_imag_acc_sqr = g12_imag_acc * g12_imag_acc;

    // det
    reg  signed [DET_WIDTH-1:0]            det;
    reg  signed [DIVOUT_Q_WIDTH-1:0]       inv_det_q;
    reg  signed [DIVOUT_F_WIDTH-1:0]       inv_det_f;
    wire signed [DIVOUT_TDATA_WIDTH-1:0]   inv_det;

    // inverse G register
    reg signed [INV_G_WIDTH-1:0] inv_g11_real;
    reg signed [INV_G_WIDTH-1:0] inv_g12_real;
    reg signed [INV_G_WIDTH-1:0] inv_g12_imag;
    reg signed [INV_G_WIDTH-1:0] inv_g22_real;

    // result elements (avoid timing violation)
    reg signed [INV_G_WIDTH-1:0] result_real_element0;
    reg signed [INV_G_WIDTH-1:0] result_real_element1;
    reg signed [INV_G_WIDTH-1:0] result_real_element2;
    reg signed [INV_G_WIDTH-1:0] result_imag_element0;
    reg signed [INV_G_WIDTH-1:0] result_imag_element1;
    reg signed [INV_G_WIDTH-1:0] result_imag_element2;

    // ==============================
    // divider input
    // ==============================

    wire signed [DIVIDEND_TDATA_WIDTH-1:0] dividend_scaled;
    wire signed [DET_WIDTH-1:0]            det_scaled;
    wire signed [DIVISOR_TDATA_WIDTH-1:0]  det_scaled_div;
    wire signed [DIVISOR_TDATA_WIDTH-1:0]  det_scaled_abs;
    wire signed [DIVISOR_TDATA_WIDTH-1:0]  det_half;
    wire signed [DIVIDEND_TDATA_WIDTH-1:0] dividend_rounded;

    // Dynamic det_shift (like software): det_shift = max(0, bit_length(|det|) - (DIVOUT_F_WIDTH+1)).
    wire [DET_WIDTH-1:0] det_mag;
    reg  [6:0]           lead_one_idx;
    wire [6:0]           det_bits;
    wire [6:0]           det_shift_dyn;
    wire [6:0]           det_shift_use;
    reg  [6:0]           det_shift_r;
    integer              lod_i;

    assign det_mag       = (det[DET_WIDTH-1]) ? (-det) : det;
    assign det_bits      = (|det_mag) ? (lead_one_idx + 1'b1) : 7'b0;
    assign det_shift_dyn = (det_bits >= TARGET_BITS) ? (det_bits - TARGET_BITS) : 7'b0;
    assign det_shift_use = USE_DYNAMIC_DET_SHIFT ? det_shift_dyn : DET_SCALE_SHIFT_7;

    // Scan low to high so the last set bit index is the leading one (MSB).
    always @(*) begin
        lead_one_idx = 0;
        for (lod_i = 0; lod_i <= DET_WIDTH - 1; lod_i = lod_i + 1)
            if (det_mag[lod_i])
                lead_one_idx = lod_i[6:0];
    end

    // Scale det: fixed or dynamic shift so divisor fits divider width.
    assign det_scaled       = det >>> det_shift_use;
    assign det_scaled_div   = det_scaled[DIVISOR_TDATA_WIDTH-1:0];
    assign det_scaled_abs   = (det_scaled_div[DIVISOR_TDATA_WIDTH-1]) ? -det_scaled_div : det_scaled_div;
    assign det_half         = det_scaled_abs >>> 1;
    // In 48-bit signed, 1<<47 is -2^47; we send this to signed divider then negate quotient for correct 1/det sign.
    assign dividend_scaled  = ($signed({{(DIVIDEND_TDATA_WIDTH - 1){1'b0}}, 1'b1})) <<< DIVIDEND_MAX_SHIFT;

    // Round-to-nearest: add +/- (|det|/2) before division.
    wire det_scaled_div_zero;

    assign det_scaled_div_zero = (det_scaled_div == 0);
    assign dividend_rounded = det_scaled_div_zero ? 0 : ((det_scaled_div[DIVISOR_TDATA_WIDTH-1]) ? (dividend_scaled - $signed(det_half)) : (dividend_scaled + $signed(det_half)));

    // ==============================
    // divider output
    // ==============================
    
    wire signed [DIVOUT_TDATA_WIDTH-1:0] inv_det_raw;
    wire inv_det_raw_zero;

    assign inv_det_raw      = ($signed(inv_det_q) <<< (DIVOUT_F_WIDTH - 1)) + $signed(inv_det_f);
    assign inv_det_raw_zero = (inv_det_q == 0) && (inv_det_f == 0);
    // Dividend is -2^47 (signed), so quotient has opposite sign of 1/det; negate to get correct inv_det.
    assign inv_det          = inv_det_raw_zero ? 0 : (-(inv_det_raw <<< (DIVIDEND_SHIFT - DIVIDEND_MAX_SHIFT)));

    // ==============================
    // FSM
    // ==============================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            state <= next_state;
        end
    end

    always @(*) begin
        case (state)
            S_IDLE:           next_state = (start_delay[LATENCY]) ? S_RD : S_IDLE;
            S_RD:             next_state = (rd_cnt == PER_FREQ - 1) ? S_PLUS : S_UPDATE_RD_ADDR;
            S_UPDATE_RD_ADDR: next_state = S_WAIT_RD_DELAY;
            S_WAIT_RD_DELAY:  next_state = S_RD;
            S_PLUS:           next_state = S_CALDET1;
            S_CALDET1:        next_state = S_CALDET2;
            S_CALDET2:        next_state = S_INVDET;
            S_INVDET:         next_state = S_SETDIV;
            S_SETDIV:         next_state = S_WAITDIV;
            S_WAITDIV:        next_state = (m_axis_dout_tvalid) ? S_CALINVG : S_WAITDIV;
            S_CALINVG:        next_state = S_CALRESULT;
            S_CALRESULT:      next_state = S_WR;
            S_WR:             next_state = (wr_cnt == PER_FREQ - 1) ? S_DONE : S_UPDATE_WR_ADDR;
            S_UPDATE_WR_ADDR: next_state = S_CALRESULT;
            S_DONE:           next_state = (freq_sample_cnt == FREQ_NUM - 1) ? S_IDLE : S_RESTART;
            S_RESTART:        next_state = S_RD;
            default:          next_state = S_IDLE;
        endcase
    end

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // bram addr
            bram_rd_addr <= BRAM_RD_ADDR_BASE;
            bram_wr_addr <= BRAM_WR_ADDR_BASE;

            // sor0 and sor1 temp register
            for (i = 0; i < MIC_NUM; i = i + 1) begin
                sor0_temp_real[i] <= 0;
                sor0_temp_imag[i] <= 0;
                sor1_temp_real[i] <= 0;
                sor1_temp_imag[i] <= 0;
            end
            flag_rd_sor1   <= 0;
            result_row1    <= 0;
            wait_rd_change <= 0;

            // G register
            g11_real_acc <= 0;
            g12_real_acc <= 0;
            g12_imag_acc <= 0;
            g22_real_acc <= 0;

            // det register
            det          <= 0;
            det_shift_r  <= 0;
            inv_det_q    <= 0;
            inv_det_f    <= 0;

            // to divider
            s_axis_dividend_tdata  <= 0;
            s_axis_dividend_tvalid <= 0;
            s_axis_divisor_tdata   <= 0;
            s_axis_divisor_tvalid  <= 0;

            // counter
            sor_cnt         <= 3'd0;
            rd_cnt          <= 4'd0;
            wr_cnt          <= 4'd0;
            freq_sample_cnt <= 9'd0;

            // result
            result_real_element0 <= 0;
            result_real_element1 <= 0;
            result_real_element2 <= 0;
            result_imag_element0 <= 0;
            result_imag_element1 <= 0;
            result_imag_element2 <= 0;
            result_bram_wr_real  <= 0;
            result_bram_wr_imag  <= 0;
            done                 <= 0;
            all_freq_finish      <= 0;
        end else begin
            case (state)
                S_IDLE: begin // state 0
                    done                 <= 0;
                    all_freq_finish      <= 0;
                    bram_rd_addr         <= BRAM_RD_ADDR_BASE;
                    bram_wr_addr         <= BRAM_WR_ADDR_BASE;
                    if (start_delay[LATENCY]) begin
                        for (i = 0; i < MIC_NUM; i = i + 1) begin
                            sor0_temp_real[i] <= 0;
                            sor0_temp_imag[i] <= 0;
                            sor1_temp_real[i] <= 0;
                            sor1_temp_imag[i] <= 0;
                        end
                        sor_cnt              <= 0;
                        rd_cnt               <= 0;
                        g11_real_acc         <= 0;
                        g12_real_acc         <= 0;
                        g12_imag_acc         <= 0;
                        g22_real_acc         <= 0;
                        inv_g11_real         <= 0;
                        inv_g12_real         <= 0;
                        inv_g12_imag         <= 0;
                        inv_g22_real         <= 0;
                        det                  <= 0;
                        det_shift_r          <= 0;
                        inv_det_q            <= 0;
                        inv_det_f            <= 0;
                        result_real_element0 <= 0;
                        result_real_element1 <= 0;
                        result_real_element2 <= 0;
                        result_imag_element0 <= 0;
                        result_imag_element1 <= 0;
                        result_imag_element2 <= 0;
                        result_bram_wr_real  <= 0;
                        result_bram_wr_imag  <= 0;
                        flag_rd_sor1         <= 0;
                        result_row1          <= 0;
                        wait_rd_change       <= 1;
                    end
                end
                S_RD: begin // state 1
                    rd_cnt <= (rd_cnt == PER_FREQ - 1) ? rd_cnt : rd_cnt + 1;
                    if (flag_rd_sor1) begin
                        sor1_temp_real[sor_cnt] <= af_bram_rd_real;
                        sor1_temp_imag[sor_cnt] <= af_bram_rd_imag;
                        g22_real_acc            <= g22_real_acc + $signed(af_bram_rd_real) * $signed(af_bram_rd_real)
                                                                + $signed(af_bram_rd_imag) * $signed(af_bram_rd_imag);
                        g12_real_acc            <= g12_real_acc + $signed(sor0_temp_real[sor_cnt]) * $signed(af_bram_rd_real)
                                                                + $signed(sor0_temp_imag[sor_cnt]) * $signed(af_bram_rd_imag);
                        g12_imag_acc            <= g12_imag_acc + $signed(sor0_temp_real[sor_cnt]) * $signed(af_bram_rd_imag)
                                                                - $signed(sor0_temp_imag[sor_cnt]) * $signed(af_bram_rd_real);
                    end else begin
                        sor0_temp_real[sor_cnt] <= af_bram_rd_real;
                        sor0_temp_imag[sor_cnt] <= af_bram_rd_imag;
                        g11_real_acc            <= g11_real_acc + $signed(af_bram_rd_real) * $signed(af_bram_rd_real)
                                                                + $signed(af_bram_rd_imag) * $signed(af_bram_rd_imag);
                    end
                end
                S_UPDATE_RD_ADDR: begin // state 2
                    sor_cnt      <= (sor_cnt == MIC_NUM - 1) ? 0 : sor_cnt + 1;
                    flag_rd_sor1 <= (sor_cnt == MIC_NUM - 1) ? ~flag_rd_sor1 : flag_rd_sor1;
                    bram_rd_addr <= bram_rd_addr + BRAM_RD_INCREASE;
                end
                S_WAIT_RD_DELAY: begin // state 3
                    // just wait
                end
                S_PLUS: begin // state 4
                    // wait_rd_change <= 0;
                    bram_rd_addr   <= (wait_rd_change) ? bram_rd_addr + BRAM_RD_INCREASE : bram_rd_addr;
                    flag_rd_sor1   <= 0;
                    rd_cnt         <= 0;
                    sor_cnt        <= 0;
                    g11_real_acc   <= g11_real_acc + $signed(LAMBDA);
                    g22_real_acc   <= g22_real_acc + $signed(LAMBDA);
                end
                S_CALDET1: begin // state 5
                    wait_rd_change <= 0;
                    det            <= g11_real_acc * g22_real_acc;
                end
                S_CALDET2: begin // state 6
                    det <= det - (g12_real_acc_sqr + g12_imag_acc_sqr);
                end
                S_INVDET: begin // state 7
                    // Only send to divider if det_scaled_div is non-zero
                    s_axis_divisor_tdata  <= det_scaled_div_zero ? 1 : det_scaled_div;
                    s_axis_dividend_tdata <= dividend_rounded;
                    det_shift_r           <= det_shift_use;
                end
                S_SETDIV: begin // state 8
                    s_axis_divisor_tvalid  <= 1;
                    s_axis_dividend_tvalid <= 1;
                end
                S_WAITDIV: begin // state 9
                    s_axis_divisor_tvalid  <= 0;
                    s_axis_dividend_tvalid <= 0;
                    if (m_axis_dout_tvalid) begin
                        inv_det_q <= m_axis_dout_tdata[DIVOUT_TDATA_WIDTH-1:DIVOUT_F_WIDTH];
                        inv_det_f <= m_axis_dout_tdata[DIVOUT_F_WIDTH-1:0];
                    end
                end
                S_CALINVG: begin // state 10
                    inv_g11_real <=  g22_real_acc * inv_det;
                    inv_g12_real <= -g12_real_acc * inv_det;
                    inv_g12_imag <= -g12_imag_acc * inv_det;
                    inv_g22_real <=  g11_real_acc * inv_det;
                end
                S_CALRESULT: begin // state 11
                    if (result_row1) begin
                        result_real_element0 <=  inv_g12_real * sor0_temp_real[sor_cnt];
                        result_real_element1 <= -inv_g12_imag * sor0_temp_imag[sor_cnt];
                        result_real_element2 <=  inv_g22_real * sor1_temp_real[sor_cnt];
                        result_imag_element0 <= -inv_g12_real * sor0_temp_imag[sor_cnt];
                        result_imag_element1 <= -inv_g12_imag * sor0_temp_real[sor_cnt];
                        result_imag_element2 <= -inv_g22_real * sor1_temp_imag[sor_cnt];
                    end else begin
                        result_real_element0 <=  inv_g11_real * sor0_temp_real[sor_cnt];
                        result_real_element1 <=  inv_g12_real * sor1_temp_real[sor_cnt];
                        result_real_element2 <=  inv_g12_imag * sor1_temp_imag[sor_cnt];
                        result_imag_element0 <= -inv_g11_real * sor0_temp_imag[sor_cnt];
                        result_imag_element1 <= -inv_g12_real * sor1_temp_imag[sor_cnt];
                        result_imag_element2 <=  inv_g12_imag * sor1_temp_real[sor_cnt];
                    end
                end
                S_WR: begin // state 12
                    wr_cnt              <= (wr_cnt == PER_FREQ - 1) ? wr_cnt : wr_cnt + 1;
                    result_bram_wr_real <= result_real_element0 + result_real_element1 + result_real_element2;
                    result_bram_wr_imag <= result_imag_element0 + result_imag_element1 + result_imag_element2;
                end
                S_UPDATE_WR_ADDR: begin // state 13
                    sor_cnt      <= (sor_cnt == MIC_NUM - 1) ? 0 : sor_cnt + 1;
                    result_row1  <= (sor_cnt == MIC_NUM - 1) ? ~result_row1 : result_row1;
                    bram_wr_addr <= bram_wr_addr + BRAM_WR_INCREASE;
                end
                S_DONE: begin // state 14
                    result_row1     <= 0;
                    sor_cnt         <= 0;
                    wr_cnt          <= 0;
                    done            <= 1;
                    bram_wr_addr    <= bram_wr_addr + BRAM_WR_INCREASE; // for next bram write start
                    if (freq_sample_cnt == FREQ_NUM - 1) begin
                        freq_sample_cnt <= 0;
                        all_freq_finish <= 1;
                    end else begin
                        freq_sample_cnt <= freq_sample_cnt + 1;
                        all_freq_finish <= 0;
                    end
                end
                S_RESTART: begin // state 15
                    for (i = 0; i < MIC_NUM; i = i + 1) begin
                        sor0_temp_real[i] <= 0;
                        sor0_temp_imag[i] <= 0;
                        sor1_temp_real[i] <= 0;
                        sor1_temp_imag[i] <= 0;
                    end
                    g11_real_acc         <= 0;
                    g12_real_acc         <= 0;
                    g12_imag_acc         <= 0;
                    g22_real_acc         <= 0;
                    inv_g11_real         <= 0;
                    inv_g12_real         <= 0;
                    inv_g12_imag         <= 0;
                    inv_g22_real         <= 0;
                    det                  <= 0;
                    inv_det_q            <= 0;
                    inv_det_f            <= 0;
                    done                 <= 0;
                    result_real_element0 <= 0;
                    result_real_element1 <= 0;
                    result_real_element2 <= 0;
                    result_imag_element0 <= 0;
                    result_imag_element1 <= 0;
                    result_imag_element2 <= 0;
                    result_bram_wr_real  <= 0;
                    result_bram_wr_imag  <= 0;
                    flag_rd_sor1         <= 0;
                    rd_cnt               <= 0;
                    bram_rd_addr         <= bram_rd_addr + BRAM_RD_INCREASE;
                end
                default: begin
                    // bram addr
                    bram_rd_addr <= BRAM_RD_ADDR_BASE;
                    bram_wr_addr <= BRAM_WR_ADDR_BASE;

                    // sor0 and sor1 temp register
                    for (i = 0; i < MIC_NUM; i = i + 1) begin
                        sor0_temp_real[i] <= 0;
                        sor0_temp_imag[i] <= 0;
                        sor1_temp_real[i] <= 0;
                        sor1_temp_imag[i] <= 0;
                    end
                    flag_rd_sor1 <= 0;
                    result_row1  <= 0;

                    // G register
                    g11_real_acc <= 0;
                    g12_real_acc <= 0;
                    g12_imag_acc <= 0;
                    g22_real_acc <= 0;

                    // det register
                    det          <= 0;
                    inv_det_q    <= 0;
                    inv_det_f    <= 0;

                    // to divider
                    s_axis_dividend_tdata  <= 0;
                    s_axis_dividend_tvalid <= 0;
                    s_axis_divisor_tdata   <= 0;
                    s_axis_divisor_tvalid  <= 0;

                    // counter
                    sor_cnt         <= 3'd0;
                    rd_cnt          <= 4'd0;
                    wr_cnt          <= 4'd0;
                    freq_sample_cnt <= 9'd0;

                    // result
                    result_real_element0 <= 0;
                    result_real_element1 <= 0;
                    result_real_element2 <= 0;
                    result_imag_element0 <= 0;
                    result_imag_element1 <= 0;
                    result_imag_element2 <= 0;
                    result_bram_wr_real  <= 0;
                    result_bram_wr_imag  <= 0;
                    done                 <= 0;
                    all_freq_finish      <= 0;
                end
            endcase
        end
    end
endmodule