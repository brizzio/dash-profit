let dados = []
const learningRate = 0.02
const optimizer = tf.train.sgd(learningRate);
let count
let tensorx, tensory
let m, b

class Dado {
    constructor(x_val, y_val, colorIndex) {
      this.x = x_val;
      this.y = y_val;
      this.color = colorIndex;
      this.distance = 0;
      this.id = 0;
      this.index = 0;
      this.descricao = "";
    }

    dist(x1, y1, x2, y2){
        this.distance = pDistance(this.x, this.y, x1, y1, x2, y2)
    }
  }


function setup(){

    var canvas = createCanvas(300, 300);
 
    // Move the canvas so it’s inside our <div id="sketch-holder">.
    canvas.parent('sketch-holder');

    
    
    m = tf.variable(tf.scalar(random(1)))
    b = tf.variable(tf.scalar(random(1)))

}

function mousePressed(){
    count = count + 1
    let x_val = map(mouseX, 0, width,0,1)
    let y_val = map(mouseY, 0, height, 1,0)

    var novo_ponto = new Dado(x_val, y_val,0)
    novo_ponto.id = count
    novo_ponto.index = count - 1
    novo_ponto.descricao = "Produto " + count


    dados.push(novo_ponto)

    
}

function draw(){

    background(0);

    stroke(255);

    strokeWeight(8);

    if (dados.length>0){
        tf.tidy(function(){
            let xs = dados.map((pt)=>pt.x)
            tensorx = tf.tensor1d(xs)

            let ys = dados.map((pt)=>pt.y)
            tensory = tf.tensor1d(ys)

           var loss_value =  optimizer.minimize(function(){
                return loss(predict(tensorx), tensory)
            },true)

            console.log("loss: " + loss_value)
        })
       
    }
   

    //calcula os valores de y para os extremos do grafico no eixo x
    const coord_xs =[0,1]
    const pred_ys = tf.tidy(()=>predict(tf.tensor1d(coord_xs)))
    const coord_ys = pred_ys.dataSync();
    pred_ys.dispose();

    const x1 = map(coord_xs[0], 0, 1, 0 , width);
    const x2 = map(coord_xs[1], 0, 1, 0 , width);

    const y1 = map(coord_ys[0], 0, 1, height, 0);
    const y2 = map(coord_ys[1], 0, 1, height, 0);
    



    dados.forEach(function(dado){

        //reverte o mapeamento para o canvas

        let coor_x = map(dado.x, 0, 1,0,width)
        let coor_y = map(dado.y, 0,1, height,0)
        //dado.dist(x1,y1,x2,y2)

        point(coor_x,coor_y)

    })

    //desenha a linha


    stroke(255);
    strokeWeight(4);
    line(x1,y1,x2,y2);

    
}



function predict(tnsr){
    
    return tnsr.mul(m).add(b)

}

function loss(pred, labels){
    return pred.sub(labels).square().mean();
}