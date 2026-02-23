`timescale 1ns / 1ps
module RTF_top_tb ();
    // Match RTF_top.v parameters
    localparam DATA_WIDTH           = 16;
    localparam INV_G_WIDTH          = 103;
    localparam BRAM_RD_ADDR_WIDTH   = 14;
    localparam BRAM_WR_ADDR_WIDTH   = 32;
    localparam BRAM_WR_WE_WIDTH     = 8;
    localparam BRAM_RD_INCREASE     = 2;
    localparam BRAM_WR_INCREASE     = 8;
    localparam MIC_NUM              = 8;
    localparam SOR_NUM              = 2;
    localparam FREQ_NUM             = 257;
    localparam PER_FREQ             = MIC_NUM * SOR_NUM;  // 16
    localparam TOTAL_NUM            = MIC_NUM * SOR_NUM * FREQ_NUM;  // 4112

    localparam PERIOD = 10;

    reg                                  clk;
    reg                                  rst_n;
    reg                                  start;
    reg  signed [DATA_WIDTH-1:0]         af_bram_rd_real;
    reg  signed [DATA_WIDTH-1:0]         af_bram_rd_imag;
    wire                                 done;
    wire                                 all_freq_finish;
    wire        [BRAM_RD_ADDR_WIDTH-1:0] bram_rd_addr;
    wire signed [INV_G_WIDTH-1:0]       result_bram_wr_real;
    wire signed [INV_G_WIDTH-1:0]       result_bram_wr_imag;
    wire        [BRAM_WR_ADDR_WIDTH-1:0] bram_wr_addr;
    wire        [BRAM_WR_WE_WIDTH-1:0]   bram_wr_we;
    wire                                 bram_wr_en;

    // Simulated BRAM for read data
    reg signed [DATA_WIDTH-1:0] bram_rd_mem_real [0:TOTAL_NUM-1];
    reg signed [DATA_WIDTH-1:0] bram_rd_mem_imag [0:TOTAL_NUM-1];

    // Simulated BRAM for write data (width matches RTF_top result_bram_wr_*)
    reg signed [INV_G_WIDTH-1:0] bram_wr_mem_real [0:TOTAL_NUM-1];
    reg signed [INV_G_WIDTH-1:0] bram_wr_mem_imag [0:TOTAL_NUM-1];
    reg                           bram_wr_valid    [0:TOTAL_NUM-1];

    integer i;

    design_2 u_design_2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .af_bram_rd_real(af_bram_rd_real),
        .af_bram_rd_imag(af_bram_rd_imag),
        .done(done),
        .all_freq_finish(all_freq_finish),
        .bram_rd_addr(bram_rd_addr),
        .result_bram_wr_real(result_bram_wr_real),
        .result_bram_wr_imag(result_bram_wr_imag),
        .bram_wr_addr(bram_wr_addr),
        .bram_wr_we(bram_wr_we),
        .bram_wr_en(bram_wr_en)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(PERIOD/2) clk = ~clk;
    end

    // Initialize BRAM read memory with simple test data
    initial begin
        for (i = 0; i < TOTAL_NUM; i = i + 1) begin
            // Simple test pattern: real part = index, imag part = index + 1
            // Scale down to fit in 16-bit signed range
            bram_rd_mem_real[i] = (i % 1000) - 500;  // Range: -500 to 499
            bram_rd_mem_imag[i] = ((i + 1) % 1000) - 500;

            // Initialize write memory
            bram_wr_mem_real[i] = 0;
            bram_wr_mem_imag[i] = 0;
            bram_wr_valid[i]    = 0;
        end
    end

    // Simulate BRAM read without extra latency
    always @(*) begin
        // Address is in bytes, convert to word index: addr / increment
        // BRAM_RD_ADDR_BASE is 0, BRAM_RD_INCREASE is 2
        if ((bram_rd_addr / BRAM_RD_INCREASE) < TOTAL_NUM) begin
            af_bram_rd_real = bram_rd_mem_real[bram_rd_addr / BRAM_RD_INCREASE];
            af_bram_rd_imag = bram_rd_mem_imag[bram_rd_addr / BRAM_RD_INCREASE];
        end else begin
            af_bram_rd_real = 0;
            af_bram_rd_imag = 0;
        end
    end
    // Simulate BRAM write
    reg [BRAM_WR_ADDR_WIDTH-1:0] wr_addr_index;
    always @(*) begin
        wr_addr_index = bram_wr_addr / BRAM_WR_INCREASE;
    end

    always @(posedge clk) begin
        if (bram_wr_en && (bram_wr_we != 0)) begin
            // Convert address to index: addr / BRAM_WR_INCREASE (8)
            // BRAM_WR_ADDR_BASE is 0
            if (bram_wr_addr >= 0 && wr_addr_index < TOTAL_NUM) begin
                bram_wr_mem_real[wr_addr_index] <= result_bram_wr_real;
                bram_wr_mem_imag[wr_addr_index] <= result_bram_wr_imag;
                bram_wr_valid[wr_addr_index]    <= 1'b1;
                $display("[%0t] BRAM Write: addr=%0d (index=%0d), real=%0d, imag=%0d",
                         $time, bram_wr_addr, wr_addr_index,
                         result_bram_wr_real, result_bram_wr_imag);
            end else if (bram_wr_addr >= 0) begin
                $display("[%0t] Warning: BRAM write address out of range: addr=%0d (index=%0d)",
                         $time, bram_wr_addr, wr_addr_index);
            end
        end
    end

    // Test sequence
    integer round;
    integer f;
    initial begin
        // Initialize
        rst_n = 0;
        start = 0;

        // Reset
        #(PERIOD * 10);
        rst_n = 1;
        #(PERIOD * 30);

        // Run two rounds; each round sweeps all FREQ_NUM freqs.
        // For each freq, send one start pulse (process PER_FREQ = 16 samples).
        for (round = 0; round < 1; round = round + 1) begin
            $display("========================================");
            $display("Round %0d: start running all %0d freqs", round, FREQ_NUM);
            $display("========================================");

            for (f = 0; f < FREQ_NUM; f = f + 1) begin
                // start pulse for this frequency
                @(posedge clk);
                start = 1'b1;
                @(posedge clk);
                start = 1'b0;

                wait (done == 1'b0);
                wait (done == 1'b1);

                // Optional: display progress every 10 freqs
                if ((f + 1) % 10 == 0) begin
                    $display("[%0t] Processed %0d/%0d frequencies", $time, f + 1, FREQ_NUM);
                end
            end

            $display("[%0t] Round %0d finished, all_freq_finish = %0d",
                     $time, round, all_freq_finish);
            #(PERIOD * 10);
        end

        // Display some write results
        $display("========================================");
        $display("Sample Write Results (first 10 entries):");
        $display("========================================");
        for (i = 0; i < 10 && i < TOTAL_NUM; i = i + 1) begin
            if (bram_wr_valid[i]) begin
                $display("  [%0d] real=%0d, imag=%0d",
                         i, bram_wr_mem_real[i], bram_wr_mem_imag[i]);
            end
        end

        #(PERIOD * 10);
        $display("========================================");
        $display("Test Complete");
        $display("========================================");
        $finish;
    end

endmodule