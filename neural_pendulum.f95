program neural_pendulum
    implicit none
    integer, parameter :: input_nodes = 4, hidden_nodes = 6, output_nodes = 4
    real, parameter :: eta = 0.01 ! Współczynnik uczenia

    real :: w_input_hidden(input_nodes, hidden_nodes)
    real :: w_hidden_output(hidden_nodes, output_nodes)
    real :: bias_hidden(hidden_nodes), bias_output(output_nodes)

    real :: inputs(input_nodes), hidden(hidden_nodes), outputs(output_nodes)
    real :: expected(output_nodes)
    integer :: i, j, epoch, max_epochs

    max_epochs = 10000 ! Liczba epok

    ! Inicjalizacja wag losowymi wartościami
    call initialize_weights()

    ! Pętla treningowa
    do epoch = 1, max_epochs
        ! Wczytaj przykładowe dane z symulacji (tu na razie losowe)
        call load_sample(inputs, expected)

        ! Propagacja w przód
        call forward_pass(inputs, hidden, outputs)

        ! Aktualizacja wag
        call backpropagate(inputs, hidden, outputs, expected)

        if (mod(epoch, 1000) == 0) then
            print *, "Epoka: ", epoch, " Błąd: ", sum((outputs - expected)**2)
        end if
    end do

    print *, "Trening zakończony!"

contains

    subroutine initialize_weights()
        do i = 1, input_nodes
            do j = 1, hidden_nodes
                w_input_hidden(i, j) = 2.0 * (rand() - 0.5) ! Losowe wagi
            end do
        end do
        do i = 1, hidden_nodes
            do j = 1, output_nodes
                w_hidden_output(i, j) = 2.0 * (rand() - 0.5)
            end do
        end do
        bias_hidden = 0.0
        bias_output = 0.0
    end subroutine initialize_weights

    subroutine forward_pass(inp, hid, out)
        real, intent(in)  :: inp(input_nodes)
        real, intent(out) :: hid(hidden_nodes), out(output_nodes)
        real :: sum_h, sum_o
        integer :: i, j

        ! Warstwa ukryta
        do j = 1, hidden_nodes
            sum_h = sum(inp * w_input_hidden(:, j)) + bias_hidden(j)
            hid(j) = 1.0 / (1.0 + exp(-sum_h)) ! Sigmoid
        end do

        ! Warstwa wyjściowa
        do j = 1, output_nodes
            sum_o = sum(hid * w_hidden_output(:, j)) + bias_output(j)
            out(j) = sum_o ! Brak aktywacji w warstwie wyjściowej
        end do
    end subroutine forward_pass

    subroutine backpropagate(inp, hid, out, exp_out)
        real, intent(in)  :: inp(input_nodes), hid(hidden_nodes), out(output_nodes), exp_out(output_nodes)
        real :: error_out(output_nodes), error_hid(hidden_nodes)
        integer :: i, j

        ! Błąd warstwy wyjściowej
        error_out = exp_out - out

        ! Korekta wag warstwa ukryta -> wyjście
        do i = 1, hidden_nodes
            do j = 1, output_nodes
                w_hidden_output(i, j) = w_hidden_output(i, j) + eta * error_out(j) * hid(i)
            end do
        end do

        ! Korekta biasów wyjściowych
        bias_output = bias_output + eta * error_out

        ! Błąd warstwy ukrytej
        do i = 1, hidden_nodes
            error_hid(i) = sum(error_out * w_hidden_output(i, :)) * hid(i) * (1.0 - hid(i))
        end do

        ! Korekta wag wejście -> ukryta
        do i = 1, input_nodes
            do j = 1, hidden_nodes
                w_input_hidden(i, j) = w_input_hidden(i, j) + eta * error_hid(j) * inp(i)
            end do
        end do

        ! Korekta biasów ukrytych
        bias_hidden = bias_hidden + eta * error_hid
    end subroutine backpropagate

    subroutine load_sample(inp, exp_out)
        real, intent(out) :: inp(input_nodes), exp_out(output_nodes)
        inp = (/ 1.0, 0.5, 0.0, 0.0 /) ! Tymczasowe wartości
        exp_out = (/ 1.01, 0.49, 0.05, 0.02 /) ! Tymczasowe wartości
    end subroutine load_sample

end program neural_pendulum

